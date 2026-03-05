"""
Document intelligence helpers:
- classify extracted OCR text into a document type (including personal documents)
- detect card-like number sequences
- validate detected card sequences with Luhn (no prediction/completion)
"""

import logging
import re
from re import Pattern
from typing import Any, ClassVar, Optional

import httpx

from .pan_candidates import luhn_ok

__all__ = ["DocumentIntelligence"]

logger = logging.getLogger("ocr-service.document-intelligence")


class DocumentIntelligence:
    """Post-OCR analysis for document type and card-number metadata."""

    _CARD_PATTERN: ClassVar[Pattern[str]] = re.compile(r"(?:\d[\s\-]*){11,19}")
    _BINLIST_URL: ClassVar[str] = "https://lookup.binlist.net/{prefix}"
    _BIN_LOOKUP_TIMEOUT_SECONDS: ClassVar[float] = 2.0
    _BIN_INFO_CACHE: ClassVar[dict[str, Optional[dict[str, Any]]]] = {}
    _BIN_CACHE_MAX_SIZE: ClassVar[int] = 512

    _MAX_TYPE_CONFIDENCE: ClassVar[float] = 0.95
    _BASE_PERSONAL_DOC_CONFIDENCE: ClassVar[float] = 0.70
    _KEYWORD_SCORE_WEIGHT: ClassVar[float] = 0.05

    _CARD_KEYWORDS: ClassVar[set[str]] = {
        "tarjeta",
        "card",
        "credito",
        "crédito",
        "debit",
        "débito",
        "visa",
        "mastercard",
        "amex",
        "diners",
        "cvv",
        "cvc",
        "exp",
        "venc",
    }
    _INVOICE_KEYWORDS: ClassVar[set[str]] = {
        "factura",
        "invoice",
        "subtotal",
        "iva",
        "ruc",
        "impuesto",
        "total",
    }
    _RECEIPT_KEYWORDS: ClassVar[set[str]] = {
        "recibo",
        "receipt",
        "merchant",
        "store",
        "autorizacion",
        "autorización",
        "terminal",
    }
    _ID_KEYWORDS: ClassVar[set[str]] = {
        "dni",
        "cedula",
        "cédula",
        "pasaporte",
        "passport",
        "identidad",
        "id",
    }

    _PASSPORT_KEYWORDS: ClassVar[set[str]] = {
        "passport",
        "pasaporte",
        "passeport",
        "reisepass",
        "mrz",
        "p<",
        "nationality",
        "nacionalidad",
        "place of birth",
    }
    _DRIVER_LICENSE_KEYWORDS: ClassVar[set[str]] = {
        "driving licence",
        "driver's license",
        "driver license",
        "licencia de conducir",
        "permis de conduire",
        "führerschein",
        "fuhrerschein",
        "driving",
        "categories",
        "vehicle",
    }
    _NATIONAL_ID_KEYWORDS: ClassVar[set[str]] = {
        "national id",
        "national identity",
        "cedula de identidad",
        "documento nacional",
        "numero de identificacion",
        "número de identificación",
        "identity card",
        "carte nationale",
        "personalausweis",
    }
    _TAX_ID_KEYWORDS: ClassVar[set[str]] = {
        "tax id",
        "taxpayer",
        "nif",
        "cif",
        "rfc",
        "cpf",
        "cnpj",
        "tin",
        "vat",
        "fiscal",
        "tributario",
        "identification number",
    }
    _UTILITY_BILL_KEYWORDS: ClassVar[set[str]] = {
        "utility",
        "electricity",
        "electric",
        "water",
        "gas",
        "internet",
        "phone bill",
        "kwh",
        "meter reading",
        "consumption",
        "lectura",
        "consumo",
        "servicio",
        "suministro",
    }
    _BANK_STATEMENT_KEYWORDS: ClassVar[set[str]] = {
        "bank statement",
        "account statement",
        "estado de cuenta",
        "extracto",
        "balance",
        "transactions",
        "transacciones",
        "opening balance",
        "closing balance",
        "debit",
        "credit",
        "iban",
        "swift",
        "routing",
    }
    _PAYSLIP_KEYWORDS: ClassVar[set[str]] = {
        "payslip",
        "pay stub",
        "salary",
        "salario",
        "nomina",
        "nómina",
        "payroll",
        "earnings",
        "deductions",
        "gross",
        "net pay",
        "employer",
        "employee id",
    }
    _EMPLOYMENT_LETTER_KEYWORDS: ClassVar[set[str]] = {
        "employment letter",
        "carta de empleo",
        "carta laboral",
        "to whom it may concern",
        "a quien corresponda",
        "employed",
        "employment",
        "position",
        "designation",
        "annual salary",
        "full time",
    }
    _RESIDENCE_PERMIT_KEYWORDS: ClassVar[set[str]] = {
        "residence permit",
        "permiso de residencia",
        "residency",
        "resident",
        "visa",
        "immigration",
        "foreign national",
        "valid for",
    }
    _MEMBERSHIP_CARD_KEYWORDS: ClassVar[set[str]] = {
        "membership",
        "member",
        "club",
        "loyalty",
        "rewards",
        "points",
        "member since",
        "member id",
    }

    @staticmethod
    def _normalize_bin_prefix(value: str) -> str:
        """Normalize BIN input and return a 6-8 digit lookup prefix."""
        digits = re.sub(r"\D", "", value or "")
        if len(digits) < 6:
            return ""
        return digits[:8] if len(digits) >= 8 else digits[:6]

    @classmethod
    def _cache_bin_info(
        cls, prefix: str, value: Optional[dict[str, Any]]
    ) -> Optional[dict[str, Any]]:
        """Store BIN lookup result in bounded in-memory cache."""
        if len(cls._BIN_INFO_CACHE) >= cls._BIN_CACHE_MAX_SIZE:
            oldest = next(iter(cls._BIN_INFO_CACHE), None)
            if oldest is not None:
                cls._BIN_INFO_CACHE.pop(oldest, None)
        cls._BIN_INFO_CACHE[prefix] = value
        return dict(value) if value else None

    @classmethod
    def fetch_bin_info(cls, bin_prefix: str) -> Optional[dict[str, Any]]:
        """
        Query BINLIST for issuer/country/type metadata using first 6-8 digits.
        This enriches metadata only; it never completes hidden PAN digits.
        """
        normalized = cls._normalize_bin_prefix(bin_prefix)
        if not normalized:
            return None

        if normalized in cls._BIN_INFO_CACHE:
            cached = cls._BIN_INFO_CACHE[normalized]
            return dict(cached) if cached else None

        try:
            response = httpx.get(
                cls._BINLIST_URL.format(prefix=normalized),
                headers={"Accept-Version": "3"},
                timeout=cls._BIN_LOOKUP_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            payload = response.json()
            info = {
                "issuer": payload.get("bank", {}).get("name") or "unknown",
                "country": payload.get("country", {}).get("name") or "unknown",
                "type": payload.get("type") or "unknown",
                "brand": payload.get("brand") or "unknown",
                "scheme": payload.get("scheme") or "unknown",
            }
            return cls._cache_bin_info(normalized, info)
        except (httpx.HTTPError, ValueError) as e:
            logger.debug("BIN lookup failed for prefix %s: %s", normalized, e)
            cls._cache_bin_info(normalized, None)
            return None

    @classmethod
    def analyze(
        cls,
        text: str,
        layout_type: str = "unknown",
        include_bin_info: bool = False,
    ) -> dict[str, Any]:
        """
        Analyze OCR text and return:
        - document_type
        - type_confidence
        - card_analysis metadata (masked candidates + Luhn validity)
        """
        candidates = cls._extract_card_candidates(text)
        card_rows = []
        for number in candidates:
            row = {
                "masked": cls._mask_number(number),
                "length": len(number),
                "luhn_valid": cls.is_valid_luhn(number),
                "brand_guess": cls.guess_card_brand(number),
            }
            if include_bin_info:
                if bin_info := cls.fetch_bin_info(number):
                    row["bin_info"] = bin_info
            card_rows.append(row)

        luhn_valid_count = sum(row["luhn_valid"] for row in card_rows)
        card_analysis = {
            "detected": len(card_rows) > 0,
            "candidate_count": len(card_rows),
            "luhn_valid_count": luhn_valid_count,
            "requires_manual_review": len(card_rows) > 0 and luhn_valid_count == 0,
            "candidates": card_rows,
        }
        document_type, type_confidence = cls._classify_document_type(
            text, layout_type, card_rows
        )
        return {
            "document_type": document_type,
            "type_confidence": type_confidence,
            "card_analysis": card_analysis,
        }

    @classmethod
    def _extract_card_candidates(cls, text: str) -> list[str]:
        """Extract unique card-like digit sequences (11-19 digits)."""
        matches = cls._CARD_PATTERN.findall(text or "")
        seen = set()
        candidates: list[str] = []
        for match in matches:
            digits = re.sub(r"\D", "", match)
            if len(digits) < 11 or len(digits) > 19:
                continue
            if digits in seen:
                continue
            seen.add(digits)
            candidates.append(digits)
        return candidates

    @staticmethod
    def is_valid_luhn(number: str) -> bool:
        """Validate a numeric string with Luhn (13-19 digits only)."""
        return number.isdigit() and 13 <= len(number) <= 19 and luhn_ok(number)

    @staticmethod
    def _mask_number(number: str) -> str:
        """Return a masked representation that keeps only the last 4 digits."""
        if len(number) <= 4:
            return number
        masked = ("*" * (len(number) - 4)) + number[-4:]
        return " ".join(masked[i : i + 4] for i in range(0, len(masked), 4)).strip()

    @staticmethod
    def _is_mastercard(number: str) -> bool:
        if len(number) != 16:
            return False
        prefix2 = int(number[:2]) if len(number) >= 2 else 0
        prefix4 = int(number[:4]) if len(number) >= 4 else 0
        return 51 <= prefix2 <= 55 or 2221 <= prefix4 <= 2720

    @staticmethod
    def _is_discover(number: str) -> bool:
        if number.startswith("6011") or number.startswith("65"):
            return True
        if len(number) >= 3 and 644 <= int(number[:3]) <= 649:
            return True
        return len(number) >= 6 and 622126 <= int(number[:6]) <= 622925

    @staticmethod
    def _is_diners(number: str) -> bool:
        if len(number) != 14:
            return False
        if len(number) >= 3 and 300 <= int(number[:3]) <= 305:
            return True
        return len(number) >= 2 and number[:2] in {"36", "38", "39"}

    @staticmethod
    def _guess_minor_brand(number: str) -> str:
        if len(number) >= 4 and 3528 <= int(number[:4]) <= 3589:
            return "jcb"
        return "unionpay" if number.startswith("62") else "unknown"

    @classmethod
    def guess_card_brand(cls, number: str) -> str:
        """Heuristic brand guess by IIN/BIN prefix."""
        if number.startswith("4") and len(number) in {13, 16, 19}:
            return "visa"
        if len(number) >= 2 and number[:2] in {"34", "37"} and len(number) == 15:
            return "amex"
        if cls._is_mastercard(number):
            return "mastercard"
        if cls._is_discover(number):
            return "discover"
        return "diners" if cls._is_diners(number) else cls._guess_minor_brand(number)

    @classmethod
    def _classify_document_type(
        cls,
        text: str,
        layout_type: str,
        card_rows: list[dict[str, Any]],
    ) -> tuple[str, float]:
        """
        Classify document category using text keywords + card metadata.

        Returns a tuple of (document_type, type_confidence) where type_confidence
        is a float in [0.0, 1.0]. When uncertain, returns 'generic_document' with
        a lower confidence score.
        """
        lower = (text or "").lower()

        def _keyword_score(keywords: set[str]) -> int:
            return sum(kw in lower for kw in keywords)

        has_card_keyword = any(keyword in lower for keyword in cls._CARD_KEYWORDS)
        has_invoice_keyword = any(keyword in lower for keyword in cls._INVOICE_KEYWORDS)
        has_receipt_keyword = any(keyword in lower for keyword in cls._RECEIPT_KEYWORDS)
        has_id_keyword = any(keyword in lower for keyword in cls._ID_KEYWORDS)
        has_valid_card = any(row.get("luhn_valid") for row in card_rows)
        has_card_candidates = len(card_rows) > 0
        max_card_len = max((row.get("length", 0) for row in card_rows), default=0)

        passport_score = _keyword_score(cls._PASSPORT_KEYWORDS)
        driver_score = _keyword_score(cls._DRIVER_LICENSE_KEYWORDS)
        national_id_score = _keyword_score(cls._NATIONAL_ID_KEYWORDS)
        tax_id_score = _keyword_score(cls._TAX_ID_KEYWORDS)
        utility_score = _keyword_score(cls._UTILITY_BILL_KEYWORDS)
        bank_stmt_score = _keyword_score(cls._BANK_STATEMENT_KEYWORDS)
        payslip_score = _keyword_score(cls._PAYSLIP_KEYWORDS)
        employment_score = _keyword_score(cls._EMPLOYMENT_LETTER_KEYWORDS)
        residence_score = _keyword_score(cls._RESIDENCE_PERMIT_KEYWORDS)
        membership_score = _keyword_score(cls._MEMBERSHIP_CARD_KEYWORDS)

        personal_doc_candidates: list[tuple[str, int]] = [
            ("passport", passport_score),
            ("driver_license", driver_score),
            ("national_id", national_id_score),
            ("tax_id", tax_id_score),
            ("utility_bill", utility_score),
            ("bank_statement", bank_stmt_score),
            ("payslip", payslip_score),
            ("employment_letter", employment_score),
            ("residence_permit", residence_score),
            ("membership_card", membership_score),
        ]
        best_personal_type, best_personal_score = max(
            personal_doc_candidates, key=lambda x: x[1]
        )

        if best_personal_score >= 2:
            confidence = min(
                cls._MAX_TYPE_CONFIDENCE,
                cls._BASE_PERSONAL_DOC_CONFIDENCE
                + best_personal_score * cls._KEYWORD_SCORE_WEIGHT,
            )
            return best_personal_type, round(confidence, 2)

        if has_invoice_keyword:
            return "invoice", 0.90
        if has_receipt_keyword:
            return "receipt", 0.88
        if has_valid_card:
            return "bank_card", 0.95
        if has_card_candidates and (has_card_keyword or max_card_len >= 11):
            return "bank_card", 0.80

        return cls._classify_fallback(
            best_personal_score, best_personal_type, has_id_keyword, layout_type
        )

    @classmethod
    def _classify_fallback(
        cls,
        best_personal_score: int,
        best_personal_type: str,
        has_id_keyword: bool,
        layout_type: str,
    ) -> tuple[str, float]:
        """Low-confidence personal doc and generic fallback classification."""
        if has_id_keyword:
            if best_personal_score == 1:
                _id_types = {"passport", "national_id", "driver_license"}
                if best_personal_type in _id_types:
                    return best_personal_type, 0.65
                return "id_document", 0.60
            return "id_document", 0.55
        if layout_type == "dense_text":
            return "statement", 0.60
        if layout_type == "large_blocks":
            return "form", 0.55
        return "generic_document", 0.40
