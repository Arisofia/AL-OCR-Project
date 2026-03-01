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

import datetime
import logging
import re
from dataclasses import dataclass
from typing import Any, Callable, Optional

__all__ = ["PersonalDocExtractor", "ExtractedField", "detect_metadata", "_luhn_valid"]

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
    # Full name label (prefer complete names when available)
    r"(?:FULL\s+NAME|NOMBRE\s+COMPLETO)[:\s]+([A-ZÁÉÍÓÚÜÑ][A-ZÁÉÍÓÚÜÑ\s\-']{4,60})",
    # MRZ name: P<COUNTRY<SURNAME<<GIVEN<< (full name block)
    r"P<[A-Z]{3}<([A-Z<]{5,44})",
    # Fallbacks: component labels (surname / given names)
    r"(?:SURNAME|APELLIDOS?|LAST\s+NAME|NOM)[:\s]+([A-ZÁÉÍÓÚÜÑ][A-ZÁÉÍÓÚÜÑ\s\-']{2,50})",
    r"(?:GIVEN\s+NAMES?|NOMBRES?|FIRST\s+NAME|PRÉNOM)[:\s]+([A-ZÁÉÍÓÚÜÑ][A-ZÁÉÍÓÚÜÑ\s\-']{2,40})",
]

_ADDRESS_PATTERNS = [
    r"(?:ADDRESS|DIRECCIÓN|DOMICILIO|ADRESSE)[:\s]+(.{10,100}?)(?:\n|$)",
    r"(?:STREET|CALLE|RUE|STRASSE)[:\s]+(.{5,80}?)(?:\n|$)",
]

_EXPIRY_PATTERNS = [
    # Full date (DD/MM/YYYY or DD/MM/YY) – matched first to avoid truncating the year
    r"(?:EXPIRY|EXPIRATION|EXP\.?|VENC\.?|VÁLIDO\s+HASTA|VALID\s+(?:THRU|UNTIL))[:\s/]+(\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4})",
    # Short MM/YY – only when NOT followed by another date component
    r"(?:EXPIRY|EXP\.?|EXPIRATION|VENC\.?|VALID\s+THRU)[:\s/]+(\d{1,2}[/.\-]\d{2,4})(?![/.\-]\d{2,4})",
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

_OPENING_BALANCE_PATTERNS = [
    r"(?:OPENING\s+BALANCE|SALDO\s+INICIAL|SOLDE\s+INITIAL|ANFANGSSALDO)[:\s]+([£$€\d,\.\s]+)",
    r"(?:BALANCE\s+BROUGHT\s+FORWARD|B/F)[:\s]+([£$€\d,\.\s]+)",
]

_CLOSING_BALANCE_PATTERNS = [
    r"(?:CLOSING\s+BALANCE|SALDO\s+FINAL|SOLDE\s+FINAL|ENDSALDO)[:\s]+([£$€\d,\.\s]+)",
    r"(?:BALANCE\s+CARRIED\s+FORWARD|C/F|FINAL\s+BALANCE)[:\s]+([£$€\d,\.\s]+)",
]

_VAT_PATTERNS = [
    r"(?:VAT|IVA|TVA|TAX\s+AMOUNT|MWST)[:\s]+([£$€\d,\.\s%]+)",
    r"(?:VALUE\s+ADDED\s+TAX)[:\s]+([£$€\d,\.\s]+)",
]

_ISSUE_DATE_PATTERNS = [
    r"(?:ISSUE\s+DATE|DATE\s+OF\s+ISSUE|ISSUED|EMISSION\s+DATE"
    r"|FECHA\s+DE\s+EMISI[OÓ]N)[:\s]+(\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4})",
    r"(?:DATE\s+ISSUED|FECHA\s+DE\s+EXPEDICI[OÓ]N)"
    r"[:\s]+(\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4})",
]

_PLACE_OF_BIRTH_PATTERNS = [
    r"(?:PLACE\s+OF\s+BIRTH|LIEU\s+DE\s+NAISSANCE|LUGAR\s+DE\s+NACIMIENTO|LUOGO\s+DI\s+NASCITA|GEBURTSORT)[:\s]+([A-ZÁÉÍÓÚÜÑ][A-ZÁÉÍÓÚÜÑA-Za-z\s\-,]{2,50})",
    r"(?:POB|NACIDO\s+EN|BORN\s+IN)[:\s]+([A-ZÁÉÍÓÚÜÑ][A-ZÁÉÍÓÚÜÑA-Za-z\s\-,]{2,50})",
]

_OUTSTANDING_AMOUNT_PATTERNS = [
    r"(?:OUTSTANDING\s+AMOUNT|AMOUNT\s+OUTSTANDING|SALDO\s+PENDIENTE|MONTANT\s+EN\s+SOUFFRANCE)[:\s]+([£$€\d,\.\s]+)",
    r"(?:OVERDUE|PAST\s+DUE)[:\s]+([£$€\d,\.\s]+)",
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
        ("place_of_birth", _PLACE_OF_BIRTH_PATTERNS, False, "medium"),
    ],
    "national_id": [
        ("full_name", _NAME_PATTERNS, False, "medium"),
        ("date_of_birth", _DATE_PATTERNS, False, "high"),
        ("document_number", _DOC_NUMBER_PATTERNS, False, "medium"),
        ("expiry_date", _EXPIRY_PATTERNS, False, "high"),
        ("nationality", _NATIONALITY_PATTERNS, False, "high"),
        ("gender", _GENDER_PATTERNS, False, "high"),
        ("address", _ADDRESS_PATTERNS, False, "low"),
        ("place_of_birth", _PLACE_OF_BIRTH_PATTERNS, False, "medium"),
    ],
    "passport": [
        ("full_name", _NAME_PATTERNS, False, "medium"),
        ("date_of_birth", _DATE_PATTERNS, False, "high"),
        ("document_number", _PASSPORT_NUMBER_PATTERNS, False, "high"),
        ("expiry_date", _EXPIRY_PATTERNS, False, "high"),
        ("nationality", _NATIONALITY_PATTERNS, False, "high"),
        ("gender", _GENDER_PATTERNS, False, "high"),
        ("mrz_data", _MRZ_PATTERNS, False, "medium"),
        ("place_of_birth", _PLACE_OF_BIRTH_PATTERNS, False, "medium"),
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
        ("opening_balance", _OPENING_BALANCE_PATTERNS, False, "medium"),
        ("closing_balance", _CLOSING_BALANCE_PATTERNS, False, "medium"),
        ("total_amount", _TOTAL_PATTERNS, False, "medium"),
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
        ("vat_amount", _VAT_PATTERNS, False, "medium"),
        ("issue_date", _ISSUE_DATE_PATTERNS, False, "medium"),
    ],
    "receipt": [
        ("total_amount", _TOTAL_PATTERNS, False, "high"),
    ],
    # Aliases: map id_card → same fields as national_id
    "id_card": [
        ("full_name", _NAME_PATTERNS, False, "medium"),
        ("date_of_birth", _DATE_PATTERNS, False, "high"),
        ("document_number", _DOC_NUMBER_PATTERNS, False, "medium"),
        ("expiry_date", _EXPIRY_PATTERNS, False, "high"),
        ("nationality", _NATIONALITY_PATTERNS, False, "high"),
        ("gender", _GENDER_PATTERNS, False, "high"),
        ("address", _ADDRESS_PATTERNS, False, "low"),
        ("place_of_birth", _PLACE_OF_BIRTH_PATTERNS, False, "medium"),
    ],
    # Aliases: map credit_card / debit_card → same fields as bank_card
    "credit_card": [
        ("card_number", _PAN_PATTERNS, True, "high"),
        ("expiry_date", _EXPIRY_PATTERNS, False, "high"),
        ("cvv", _CVV_PATTERNS, True, "high"),
        ("cardholder_name", _NAME_PATTERNS, False, "medium"),
    ],
    "debit_card": [
        ("card_number", _PAN_PATTERNS, True, "high"),
        ("expiry_date", _EXPIRY_PATTERNS, False, "high"),
        ("cvv", _CVV_PATTERNS, True, "high"),
        ("cardholder_name", _NAME_PATTERNS, False, "medium"),
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
# Pattern-aware confidence validators (continuous learning layer)
# ---------------------------------------------------------------------------
# Design: adding a new field validation rule is purely a configuration exercise
# — define a validator function and register it in _FIELD_VALIDATORS.
#
# Each validator receives (normalized_value, raw_ocr_value) and returns:
#   (confidence_override, advisory_note)
#   confidence_override  – "high"/"medium"/"low", or None to keep existing level
#   advisory_note        – human-readable string (positive or warning), or None
#
# This registry is the "continuous learning" representation for this codebase:
# domain knowledge about valid formats (card lengths, Luhn, expiry ranges) is
# encoded here in a centrally managed, extendable configuration map rather than
# scattered across pattern strings.


def _luhn_valid(number: str) -> bool:
    """Return True when *number* (digits-only string) satisfies the Luhn algorithm."""
    if not number.isdigit() or not (13 <= len(number) <= 19):
        return False
    total = 0
    for i, ch in enumerate(reversed(number)):
        d = int(ch)
        if i % 2 == 1:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
    return total % 10 == 0


def _validate_pan(
    value: str, raw: str
) -> tuple[Optional[str], Optional[str]]:
    """
    Validate a card PAN (Primary Account Number).

    Checks (in order):
    1. Stripped value must be digits only (no mixed alpha/symbol chars).
    2. Length must be 13-19 digits (ISO 7812 range).
    3. Luhn algorithm must pass.

    Returns:
        ("high", positive note)   – all checks pass
        ("low",  warning note)    – any check fails
    """
    digits = re.sub(r"\D", "", raw)
    if not digits:
        return "low", (
            "card_number contains non-digit characters; value likely misread"
        )
    if not (13 <= len(digits) <= 19):
        return "low", (
            f"card_number digit count ({len(digits)}) is outside expected "
            "range 13-19; verify manually"
        )
    if _luhn_valid(digits):
        return "high", "Luhn check passed; confidence boosted to high"
    return "low", "Luhn check failed; card number likely misread – verify manually"


def _validate_expiry_date(
    value: str, _raw: str
) -> tuple[Optional[str], Optional[str]]:
    """
    Validate an expiry date field.

    Accepts formats after separator normalisation (separators become ``-``):
    - ``MM-YY``      – card short expiry (e.g. ``12-26``)
    - ``MM-YYYY``    – medium expiry (e.g. ``12-2026``)
    - ``DD-MM-YYYY`` / ``DD-MM-YY`` – document expiry (e.g. ``25-09-2030``)

    Adjusts confidence:
    - Valid format + month 1-12 + year within allowed range → ``"high"``
    - Month outside 1-12                                   → ``"low"`` + warning
    - Unrecognised format                                   → ``None`` (keep base)
    """
    now = datetime.date.today()
    # Allow documents up to 10 years expired (historical / archival use-cases)
    cutoff_year = now.year - 10

    # MM-YY  (e.g. "12-26")
    m = re.match(r"^(\d{1,2})-(\d{2})$", value)
    if m:
        month, year = int(m.group(1)), 2000 + int(m.group(2))
        if not (1 <= month <= 12):
            return "low", "Expiry date has invalid month (must be 01-12); verify manually"
        if year >= cutoff_year:
            return "high", "Expiry date format valid (MM/YY)"
        return None, None  # Very old; keep base confidence

    # MM-YYYY  (e.g. "12-2026")
    m = re.match(r"^(\d{1,2})-(\d{4})$", value)
    if m:
        month, year = int(m.group(1)), int(m.group(2))
        if not (1 <= month <= 12):
            return "low", "Expiry date has invalid month (must be 01-12); verify manually"
        if year >= cutoff_year:
            return "high", "Expiry date format valid (MM/YYYY)"
        return None, None

    # DD-MM-YYYY or DD-MM-YY  (e.g. "25-09-2030")
    m = re.match(r"^(\d{1,2})-(\d{1,2})-(\d{2,4})$", value)
    if m:
        day, month = int(m.group(1)), int(m.group(2))
        year_s = m.group(3)
        year = int(year_s) if len(year_s) == 4 else 2000 + int(year_s)
        if not (1 <= month <= 12):
            return "low", "Expiry date has invalid month (must be 01-12); verify manually"
        if not (1 <= day <= 31):
            return "low", "Expiry date has invalid day (must be 01-31); verify manually"
        if year >= cutoff_year:
            return "high", "Expiry date format valid (DD/MM/YYYY)"
        return None, None

    return None, None  # Unrecognised format; keep existing confidence


# Registry: maps field_name → validator callable
# To add a new validation rule, add an entry here; the execution loop in
# extract() applies all registered validators automatically.
_FIELD_VALIDATORS: dict[
    str, Callable[[str, str], tuple[Optional[str], Optional[str]]]
] = {
    "card_number": _validate_pan,
    "expiry_date": _validate_expiry_date,
}


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

            # Post-extraction format/algorithm validation (Luhn, expiry range, etc.)
            # Produces field-specific confidence adjustments and advisory notes.
            validator_note: Optional[str] = None
            validator = _FIELD_VALIDATORS.get(field_name)
            if validator:
                conf_override, validator_note = validator(normalized, raw_value)
                if conf_override is not None:
                    confidence = conf_override
                if validator_note is not None:
                    warnings.append(validator_note)

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

            if confidence == "low" and validator_note is None:
                # Skip generic warning when a validator already emitted a specific note
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
