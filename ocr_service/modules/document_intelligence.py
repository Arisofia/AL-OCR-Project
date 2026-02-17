"""
Document intelligence helpers:
- classify extracted OCR text into a document type
- detect card-like number sequences
- validate detected card sequences with Luhn (no prediction/completion)
"""

import logging
import re
from re import Pattern
from typing import Any, ClassVar, Optional

import httpx

__all__ = ["DocumentIntelligence"]

logger = logging.getLogger("ocr-service.document-intelligence")


class DocumentIntelligence:
    """Post-OCR analysis for document type and card-number metadata."""

    _CARD_PATTERN: ClassVar[Pattern[str]] = re.compile(r"(?:\d[\s\-]*){11,19}")
    _BINLIST_URL: ClassVar[str] = "https://lookup.binlist.net/{prefix}"
    _BIN_LOOKUP_TIMEOUT_SECONDS: ClassVar[float] = 2.0
    _BIN_INFO_CACHE: ClassVar[dict[str, Optional[dict[str, Any]]]] = {}
    _BIN_CACHE_MAX_SIZE: ClassVar[int] = 512

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
        - card_analysis metadata (masked candidates + Luhn validity)
        """
        candidates = cls._extract_card_candidates(text)
        card_rows = []
        for number in candidates:
            row = {
                "masked": cls._mask_number(number),
                "length": len(number),
                "luhn_valid": cls._is_valid_luhn(number),
                "brand_guess": cls._guess_card_brand(number),
            }
            if include_bin_info:
                bin_info = cls.fetch_bin_info(number)
                if bin_info:
                    row["bin_info"] = bin_info
            card_rows.append(row)

        luhn_valid_count = sum(1 for row in card_rows if row["luhn_valid"])
        card_analysis = {
            "detected": len(card_rows) > 0,
            "candidate_count": len(card_rows),
            "luhn_valid_count": luhn_valid_count,
            "requires_manual_review": len(card_rows) > 0 and luhn_valid_count == 0,
            "candidates": card_rows,
        }
        document_type = cls._classify_document_type(text, layout_type, card_rows)
        return {
            "document_type": document_type,
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
    def _is_valid_luhn(number: str) -> bool:
        """Validate a numeric string with Luhn (13-19 digits only)."""
        if not number.isdigit() or len(number) < 13 or len(number) > 19:
            return False
        total = 0
        for index, char in enumerate(reversed(number)):
            digit = int(char)
            if index % 2 == 1:
                digit *= 2
                if digit > 9:
                    digit -= 9
            total += digit
        return total % 10 == 0

    @staticmethod
    def _mask_number(number: str) -> str:
        """Return a masked representation that keeps only the last 4 digits."""
        if len(number) <= 4:
            return number
        masked = ("*" * (len(number) - 4)) + number[-4:]
        return " ".join(masked[i : i + 4] for i in range(0, len(masked), 4)).strip()

    @staticmethod
    def _guess_card_brand(number: str) -> str:
        """Heuristic brand guess by IIN/BIN prefix."""
        if number.startswith("4") and len(number) in {13, 16, 19}:
            return "visa"
        if len(number) >= 2 and number[:2] in {"34", "37"} and len(number) == 15:
            return "amex"
        if len(number) >= 2 and 51 <= int(number[:2]) <= 55 and len(number) == 16:
            return "mastercard"
        if len(number) >= 4 and 2221 <= int(number[:4]) <= 2720 and len(number) == 16:
            return "mastercard"
        if number.startswith("6011") or number.startswith("65"):
            return "discover"
        if len(number) >= 3 and 644 <= int(number[:3]) <= 649:
            return "discover"
        if len(number) >= 6 and 622126 <= int(number[:6]) <= 622925:
            return "discover"
        if len(number) >= 3 and 300 <= int(number[:3]) <= 305 and len(number) == 14:
            return "diners"
        if len(number) >= 2 and number[:2] in {"36", "38", "39"} and len(number) == 14:
            return "diners"
        if len(number) >= 4 and 3528 <= int(number[:4]) <= 3589:
            return "jcb"
        if number.startswith("62"):
            return "unionpay"
        return "unknown"

    @classmethod
    def _classify_document_type(
        cls,
        text: str,
        layout_type: str,
        card_rows: list[dict[str, Any]],
    ) -> str:
        """Classify document category using text keywords + card metadata."""
        lower = (text or "").lower()
        has_card_keyword = any(keyword in lower for keyword in cls._CARD_KEYWORDS)
        has_invoice_keyword = any(keyword in lower for keyword in cls._INVOICE_KEYWORDS)
        has_receipt_keyword = any(keyword in lower for keyword in cls._RECEIPT_KEYWORDS)
        has_id_keyword = any(keyword in lower for keyword in cls._ID_KEYWORDS)
        has_valid_card = any(row.get("luhn_valid") for row in card_rows)
        has_card_candidates = len(card_rows) > 0
        max_card_len = max((row.get("length", 0) for row in card_rows), default=0)

        if has_invoice_keyword:
            return "invoice"
        if has_receipt_keyword:
            return "receipt"
        if has_id_keyword:
            return "id_document"
        if has_valid_card:
            return "bank_card"
        if has_card_candidates and (has_card_keyword or max_card_len >= 11):
            return "bank_card"
        if layout_type == "dense_text":
            return "statement"
        if layout_type == "large_blocks":
            return "form"
        return "generic"
