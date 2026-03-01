"""
Personal document field extractor.

Extracts structured fields from OCR text for various personal document types:
  - id_document / national_id / id_card
  - passport
  - driver_license
  - tax_id
  - bank_card (sensitive fields masked)
  - utility_bill
  - bank_statement
  - payslip
  - employment_letter
  - generic_document

Design goals:
- Country-configurable patterns live in FIELD_PATTERNS (not scattered in code).
- Adding a new country or document variant is a matter of extending the config,
  not refactoring this module.
- Sensitive fields (PAN, CVV, full card numbers) are masked in the response and
  must never appear in logs.
- Each extracted field carries raw_ocr, normalized value, and a confidence level.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional

__all__ = ["PersonalDocExtractor", "ExtractedField", "detect_metadata"]

logger = logging.getLogger("ocr-service.personal-doc-extractor")

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ExtractedField:
    """Represents a single field extracted from a document."""

    name: str
    value: str
    raw_ocr: Optional[str] = None
    confidence_level: str = "low"  # "high", "medium", or "low"


# ---------------------------------------------------------------------------
# Country / document-type pattern configuration
# ---------------------------------------------------------------------------
# Patterns are keyed by field name.  Each entry is a list of regex patterns
# tried in order; the first match wins.  This structure makes adding new
# countries or variants purely a configuration exercise.

_DATE_PATTERNS = [
    # ISO: 1990-05-12
    r"\b(\d{4}[-/]\d{2}[-/]\d{2})\b",
    # DMY: 12/05/1990 or 12-05-1990 or 12.05.1990
    r"\b(\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4})\b",
    # MDY (US): 05/12/1990
    r"\b(\d{1,2}[/]\d{1,2}[/]\d{2,4})\b",
]

_DOC_NUMBER_PATTERNS = [
    # Alphanumeric IDs like "A1234567" or "DNI 12345678X"
    r"\b(?:DNI|NIE|NIF|ID|No\.?|NUM\.?|CÉDULA|CEDULA|DOC)[\s:#]*([A-Z0-9]{6,20})\b",
    r"\b([A-Z]{1,3}\s?\d{6,12})\b",
    r"\b(\d{7,12}[A-Z]?)\b",
]

_PASSPORT_NUMBER_PATTERNS = [
    r"\b([A-Z]{1,2}\d{6,8})\b",
    r"\b([A-Z0-9]{8,9})\b",
]

# MRZ lines: two lines of 44 (TD3/passport) or 30/36 characters
_MRZ_PATTERNS = [
    r"([A-Z0-9<]{30,44}\n[A-Z0-9<]{30,44})",
    r"([A-Z0-9<]{30,44})",
]

_NAME_PATTERNS = [
    # After common labels
    r"(?:SURNAME|APELLIDOS?|LAST\s+NAME|NOM)[:\s]+([A-ZÁÉÍÓÚÜÑ][A-ZÁÉÍÓÚÜÑ\s\-']{2,50})",
    r"(?:GIVEN\s+NAMES?|NOMBRES?|FIRST\s+NAME|PRÉNOM)[:\s]+([A-ZÁÉÍÓÚÜÑ][A-ZÁÉÍÓÚÜÑ\s\-']{2,40})",
    # MRZ name: P<COUNTRY<SURNAME<<GIVEN<<
    r"P<[A-Z]{3}<([A-Z<]{5,44})",
    # Full name label
    r"(?:FULL\s+NAME|NOMBRE\s+COMPLETO)[:\s]+([A-ZÁÉÍÓÚÜÑ][A-ZÁÉÍÓÚÜÑ\s\-']{4,60})",
]

_ADDRESS_PATTERNS = [
    r"(?:ADDRESS|DIRECCIÓN|DOMICILIO|ADRESSE)[:\s]+(.{10,100}?)(?:\n|$)",
    r"(?:STREET|CALLE|RUE|STRASSE)[:\s]+(.{5,80}?)(?:\n|$)",
]

_EXPIRY_PATTERNS = [
    r"(?:EXPIRY|EXP\.?|EXPIRATION|VENC\.?|VÁLIDO\s+HASTA|VALID\s+THRU)[:\s/]+(\d{1,2}[/.\-]\d{2,4})",
    r"(?:VALID\s+UNTIL|VÁLIDO\s+HASTA)[:\s]+(\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4})",
]

_NATIONALITY_PATTERNS = [
    r"(?:NATIONALITY|NACIONALIDAD|NATIONALITÉ)[:\s]+([A-Z]{2,3}|[A-Za-z]{4,20})",
]

_GENDER_PATTERNS = [
    r"(?:SEX|SEXO|GENRE|GESCHLECHT)[:\s]+([MFX])",
    r"\b(MALE|FEMALE|MASCULINO|FEMENINO)\b",
]

_TAX_NUMBER_PATTERNS = [
    # Spain NIF/NIE, Mexico RFC, Brazil CPF/CNPJ, generic
    r"(?:NIF|NIE|RFC|CPF|CNPJ|TIN|VAT|TAX\s+ID)[:\s#]*([A-Z0-9\-\.]{6,20})",
    r"\b(\d{3}[.\-]\d{3}[.\-]\d{3}[.\-]\d{1,2})\b",  # Brazil CPF: 000.000.000-00
    r"\b([A-Z]{4}\d{6}[A-Z0-9]{3})\b",  # Mexico RFC
]

_PAN_PATTERNS = [
    r"\b(\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4})\b",
    r"\b(\d{4}[\s\-]?\d{6}[\s\-]?\d{5})\b",  # Amex 15-digit
]

_CVV_PATTERNS = [
    r"(?:CVV|CVC|CVV2|CVC2|CSC)[:\s]+(\d{3,4})",
]

_ACCOUNT_NUMBER_PATTERNS = [
    r"(?:ACCOUNT\s+(?:NUMBER|NO\.?)|CUENTA|IBAN|NUMÉRO\s+DE\s+COMPTE)[:\s#]*([A-Z0-9\s\-]{8,34})",
    r"\b(IBAN[\s:]+[A-Z]{2}\d{2}[A-Z0-9\s]{11,29})\b",
]

_EMPLOYER_PATTERNS = [
    r"(?:EMPLOYER|EMPRESA|COMPANY|EMPLOYEUR)[:\s]+(.{3,60}?)(?:\n|$)",
]

_SALARY_PATTERNS = [
    r"(?:SALARY|SALARIO|GROSS\s+PAY|NET\s+PAY|SALAIRE)[:\s]+([£$€\d,\.\s]+)",
    r"(?:TOTAL\s+EARNINGS|TOTAL\s+SALARIO)[:\s]+([£$€\d,\.\s]+)",
]

_PERIOD_PATTERNS = [
    r"(?:PAY\s+PERIOD|PERIODO|PERIOD|FOR\s+THE\s+MONTH\s+OF)[:\s]+(.{3,40}?)(?:\n|$)",
    r"(?:STATEMENT\s+PERIOD|FROM|DESDE)[:\s]+(\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4})",
]

_TOTAL_PATTERNS = [
    r"(?:TOTAL\s+AMOUNT|TOTAL|IMPORTE\s+TOTAL|MONTANT\s+TOTAL)[:\s]+([£$€\d,\.\s]+)",
    r"(?:AMOUNT\s+DUE|SALDO\s+TOTAL|BALANCE\s+DUE)[:\s]+([£$€\d,\.\s]+)",
]

# ---------------------------------------------------------------------------
# Field definitions per document type
# ---------------------------------------------------------------------------
# Each entry: (field_name, patterns_list, is_sensitive, base_confidence)
#   is_sensitive: if True the value is masked before returning
#   base_confidence: "high"/"medium"/"low" – reduced when reconstruction needed

_FieldDef = tuple[str, list[str], bool, str]

FIELD_DEFINITIONS: dict[str, list[_FieldDef]] = {
    "id_document": [
        ("full_name", _NAME_PATTERNS, False, "medium"),
        ("date_of_birth", _DATE_PATTERNS, False, "high"),
        ("document_number", _DOC_NUMBER_PATTERNS, False, "medium"),
        ("expiry_date", _EXPIRY_PATTERNS, False, "high"),
        ("nationality", _NATIONALITY_PATTERNS, False, "high"),
        ("gender", _GENDER_PATTERNS, False, "high"),
        ("address", _ADDRESS_PATTERNS, False, "low"),
    ],
    "national_id": [
        ("full_name", _NAME_PATTERNS, False, "medium"),
        ("date_of_birth", _DATE_PATTERNS, False, "high"),
        ("document_number", _DOC_NUMBER_PATTERNS, False, "medium"),
        ("expiry_date", _EXPIRY_PATTERNS, False, "high"),
        ("nationality", _NATIONALITY_PATTERNS, False, "high"),
        ("gender", _GENDER_PATTERNS, False, "high"),
        ("address", _ADDRESS_PATTERNS, False, "low"),
    ],
    "passport": [
        ("full_name", _NAME_PATTERNS, False, "medium"),
        ("date_of_birth", _DATE_PATTERNS, False, "high"),
        ("document_number", _PASSPORT_NUMBER_PATTERNS, False, "high"),
        ("expiry_date", _EXPIRY_PATTERNS, False, "high"),
        ("nationality", _NATIONALITY_PATTERNS, False, "high"),
        ("gender", _GENDER_PATTERNS, False, "high"),
        ("mrz_data", _MRZ_PATTERNS, False, "medium"),
    ],
    "driver_license": [
        ("full_name", _NAME_PATTERNS, False, "medium"),
        ("date_of_birth", _DATE_PATTERNS, False, "high"),
        ("document_number", _DOC_NUMBER_PATTERNS, False, "medium"),
        ("expiry_date", _EXPIRY_PATTERNS, False, "high"),
        ("address", _ADDRESS_PATTERNS, False, "low"),
    ],
    "tax_id": [
        ("full_name", _NAME_PATTERNS, False, "medium"),
        ("tax_number", _TAX_NUMBER_PATTERNS, False, "high"),
        ("date_of_birth", _DATE_PATTERNS, False, "medium"),
        ("address", _ADDRESS_PATTERNS, False, "low"),
    ],
    "bank_card": [
        # PAN is sensitive — masked to last 4 digits in the response
        ("card_number", _PAN_PATTERNS, True, "high"),
        ("expiry_date", _EXPIRY_PATTERNS, False, "high"),
        # CVV is highly sensitive — omitted from response entirely
        ("cvv", _CVV_PATTERNS, True, "high"),
        ("cardholder_name", _NAME_PATTERNS, False, "medium"),
    ],
    "utility_bill": [
        ("full_name", _NAME_PATTERNS, False, "medium"),
        ("address", _ADDRESS_PATTERNS, False, "medium"),
        ("account_number", _ACCOUNT_NUMBER_PATTERNS, False, "medium"),
        ("period", _PERIOD_PATTERNS, False, "high"),
        ("total_amount", _TOTAL_PATTERNS, False, "high"),
    ],
    "bank_statement": [
        ("full_name", _NAME_PATTERNS, False, "medium"),
        ("address", _ADDRESS_PATTERNS, False, "medium"),
        ("account_number", _ACCOUNT_NUMBER_PATTERNS, False, "medium"),
        ("period", _PERIOD_PATTERNS, False, "high"),
        ("opening_balance", _TOTAL_PATTERNS[:1], False, "medium"),
    ],
    "statement": [
        ("full_name", _NAME_PATTERNS, False, "medium"),
        ("account_number", _ACCOUNT_NUMBER_PATTERNS, False, "medium"),
        ("period", _PERIOD_PATTERNS, False, "high"),
        ("total_amount", _TOTAL_PATTERNS, False, "medium"),
    ],
    "payslip": [
        ("full_name", _NAME_PATTERNS, False, "medium"),
        ("employer", _EMPLOYER_PATTERNS, False, "medium"),
        ("salary", _SALARY_PATTERNS, False, "high"),
        ("period", _PERIOD_PATTERNS, False, "high"),
    ],
    "employment_letter": [
        ("full_name", _NAME_PATTERNS, False, "medium"),
        ("employer", _EMPLOYER_PATTERNS, False, "medium"),
        ("salary", _SALARY_PATTERNS, False, "low"),
    ],
    "residence_permit": [
        ("full_name", _NAME_PATTERNS, False, "medium"),
        ("date_of_birth", _DATE_PATTERNS, False, "high"),
        ("document_number", _DOC_NUMBER_PATTERNS, False, "medium"),
        ("expiry_date", _EXPIRY_PATTERNS, False, "high"),
        ("nationality", _NATIONALITY_PATTERNS, False, "high"),
    ],
    "membership_card": [
        ("full_name", _NAME_PATTERNS, False, "medium"),
        ("document_number", _DOC_NUMBER_PATTERNS, False, "medium"),
        ("expiry_date", _EXPIRY_PATTERNS, False, "high"),
    ],
    # Aliases for extended document types that share the same definitions
    "invoice": [
        ("full_name", _NAME_PATTERNS, False, "low"),
        ("total_amount", _TOTAL_PATTERNS, False, "high"),
        ("account_number", _ACCOUNT_NUMBER_PATTERNS, False, "medium"),
        ("period", _PERIOD_PATTERNS, False, "medium"),
    ],
    "receipt": [
        ("total_amount", _TOTAL_PATTERNS, False, "high"),
    ],
}

# Document types that are treated as generic (minimal field extraction)
_GENERIC_DOC_TYPES = {"generic", "generic_document", "form", "unknown"}

# Sensitive field names — value masked, never logged in plain text
_SENSITIVE_FIELDS = {"card_number", "cvv", "pan", "cvc", "cvv2", "cvc2"}

# Fields that are completely omitted from API responses (too sensitive)
_OMIT_FROM_RESPONSE = {"cvv", "cvc", "cvv2", "cvc2"}


# ---------------------------------------------------------------------------
# Language / country detection helpers
# ---------------------------------------------------------------------------

_LANG_HINTS: list[tuple[re.Pattern, str, str]] = [
    # (pattern, language_code, country_code)
    (re.compile(r"\b(cpf|cnpj|rg)\b", re.I), "pt", "BR"),
    (re.compile(r"\b(rfc|curp|ine)\b", re.I), "es", "MX"),
    (re.compile(r"\b(dni|nie)\b", re.I), "es", "ES"),
    (re.compile(r"\b(ced[uú]la)\b", re.I), "es", "CO"),
    (re.compile(r"\b(apellido|nombre|fecha|número)\b", re.I), "es", ""),
    (re.compile(r"\b(nom|prénom|date\s+de\s+naissance)\b", re.I), "fr", "FR"),
    (re.compile(r"\b(vorname|nachname|geburtsdatum)\b", re.I), "de", "DE"),
    (re.compile(r"\b(nome|data\s+di\s+nascita)\b", re.I), "it", "IT"),
    (re.compile(r"\b(national\s+insurance|ni\s+number)\b", re.I), "en", "GB"),
    (re.compile(r"\b(social\s+security|ssn)\b", re.I), "en", "US"),
]


def _detect_language_and_country(text: str) -> tuple[str, str]:
    """Return (language_code, country_code) based on text cues."""
    for pattern, lang, country in _LANG_HINTS:
        if pattern.search(text):
            return lang, country
    return "en", ""


# ---------------------------------------------------------------------------
# Masking helpers
# ---------------------------------------------------------------------------


def _mask_pan(value: str) -> str:
    """Keep only the last 4 digits of a PAN; mask the rest."""
    digits = re.sub(r"\D", "", value)
    if len(digits) <= 4:
        return value
    masked_digits = "*" * (len(digits) - 4) + digits[-4:]
    # Rebuild with spaces every 4 chars
    return " ".join(
        masked_digits[i : i + 4] for i in range(0, len(masked_digits), 4)
    ).strip()


# ---------------------------------------------------------------------------
# Core extractor
# ---------------------------------------------------------------------------


class PersonalDocExtractor:
    """
    Extracts structured fields from OCR text for personal documents.

    Usage::

        extractor = PersonalDocExtractor()
        fields, warnings = extractor.extract(text, document_type)

    The returned fields list contains :class:`ExtractedField` instances;
    sensitive values are already masked.  ``warnings`` is a list of
    human-readable strings describing low-confidence or reconstructed fields.
    """

    def extract(
        self,
        text: str,
        document_type: str,
    ) -> tuple[list[ExtractedField], list[str]]:
        """
        Extract fields from *text* for the given *document_type*.

        Returns:
            (fields, warnings) where fields is a list of ExtractedField and
            warnings is a list of advisory strings for the caller.
        """
        fields: list[ExtractedField] = []
        warnings: list[str] = []

        if document_type in _GENERIC_DOC_TYPES:
            return fields, warnings

        definitions = FIELD_DEFINITIONS.get(document_type)
        if definitions is None:
            # Attempt to fall back to id_document extraction for unknown personal docs
            logger.debug(
                "No field definitions for document_type=%r; using id_document fallback",
                document_type,
            )
            definitions = FIELD_DEFINITIONS.get("id_document", [])

        for field_name, patterns, is_sensitive, base_confidence in definitions:
            # Fields that must never appear in the API response
            if field_name in _OMIT_FROM_RESPONSE:
                continue

            raw_value = self._try_patterns(text, patterns)
            if raw_value is None:
                continue

            normalized = self._normalize(field_name, raw_value)
            confidence = self._adjust_confidence(
                base_confidence, raw_value, normalized
            )

            # Mask sensitive values before returning
            display_value = normalized
            if is_sensitive or field_name in _SENSITIVE_FIELDS:
                display_value = _mask_pan(normalized)
                logger.info(
                    "Sensitive field '%s' extracted and masked (last 4 preserved). "
                    "Raw value NOT logged.",
                    field_name,
                )

            ef = ExtractedField(
                name=field_name,
                value=display_value,
                raw_ocr=raw_value if not is_sensitive else "[REDACTED]",
                confidence_level=confidence,
            )
            fields.append(ef)

            if confidence == "low":
                warnings.append(
                    f"{field_name} extracted with low confidence; "
                    "verify manually"
                )
            elif confidence == "medium" and normalized != raw_value:
                warnings.append(
                    f"{field_name} partially reconstructed from OCR output; "
                    "verify manually"
                )

        return fields, warnings

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _try_patterns(text: str, patterns: list[str]) -> Optional[str]:
        """Return the first match found in *text* for any pattern in *patterns*."""
        for pat in patterns:
            try:
                m = re.search(pat, text, re.IGNORECASE | re.MULTILINE)
                if m:
                    return m.group(1).strip() if m.lastindex else m.group(0).strip()
            except re.error as exc:
                logger.debug("Regex error for pattern %r: %s", pat, exc)
        return None

    @staticmethod
    def _normalize(field_name: str, raw: str) -> str:
        """Apply field-specific normalization to a raw OCR match."""
        value = raw.strip()

        if "date" in field_name or field_name == "expiry_date":
            # Normalize separators → ISO-like where possible
            value = re.sub(r"[./]", "-", value)

        if field_name == "full_name":
            # Remove MRZ filler characters and collapse whitespace
            value = re.sub(r"<+", " ", value)
            value = re.sub(r"\s+", " ", value).upper().strip()

        if field_name == "mrz_data":
            # Keep MRZ compact (single line with newline if two lines present)
            value = re.sub(r"[^\w<\n]", "", value).strip()

        if field_name == "nationality":
            value = value.upper().strip()

        if field_name == "gender":
            # Normalize to single letter
            mapping = {
                "MALE": "M",
                "FEMALE": "F",
                "MASCULINO": "M",
                "FEMENINO": "F",
            }
            value = mapping.get(value.upper(), value.upper()[:1])

        return value

    @staticmethod
    def _adjust_confidence(
        base_confidence: str,
        raw: str,
        normalized: str,
    ) -> str:
        """
        Downgrade confidence when reconstruction artifacts are present
        (e.g. '?' markers or unusual character mix).
        """
        if "?" in raw:
            # '?' inserted by OCR engine to indicate uncertain character
            return "low"
        suspicious_chars = sum(1 for c in raw if c in "!|")
        if suspicious_chars >= 2:
            return "low"
        if raw != normalized and base_confidence == "high":
            return "medium"
        return base_confidence


# ---------------------------------------------------------------------------
# Module-level helper used by the router
# ---------------------------------------------------------------------------


def detect_metadata(text: str) -> dict[str, Any]:
    """Return a metadata dict with language and country guesses."""
    lang, country = _detect_language_and_country(text)
    return {
        "language_guess": lang,
        "country_guess": country or None,
    }
