"""
Tests for PersonalDocExtractor and enhanced DocumentIntelligence classification.
"""

import pytest

from ocr_service.modules.document_intelligence import DocumentIntelligence
from ocr_service.modules.personal_doc_extractor import (
    PersonalDocExtractor,
    _luhn_valid,
    detect_metadata,
)

# ---------------------------------------------------------------------------
# Enhanced DocumentIntelligence classification tests
# ---------------------------------------------------------------------------


def test_analyze_returns_type_confidence():
    """analyze() must return type_confidence as a float."""
    result = DocumentIntelligence.analyze("Hello world")
    assert "type_confidence" in result
    assert isinstance(result["type_confidence"], float)
    assert 0.0 <= result["type_confidence"] <= 1.0


def test_classify_passport():
    text = "PASSPORT\nNationality: GBR\nPlace of birth: London\n"
    result = DocumentIntelligence.analyze(text)
    assert result["document_type"] == "passport"
    assert result["type_confidence"] >= 0.65


def test_classify_driver_license():
    text = "Driving Licence\nCategories: B, C\nVehicle: car\n"
    result = DocumentIntelligence.analyze(text)
    assert result["document_type"] == "driver_license"
    assert result["type_confidence"] >= 0.65


def test_classify_national_id():
    text = (
        "NATIONAL IDENTITY CARD\n"
        "Documento Nacional de Identidad\n"
        "Número de identificación: 12345678X"
    )
    result = DocumentIntelligence.analyze(text)
    assert result["document_type"] == "national_id"
    assert result["type_confidence"] >= 0.65


def test_classify_tax_id():
    text = "NIF: 12345678Z\nFiscal identification number\n"
    result = DocumentIntelligence.analyze(text)
    assert result["document_type"] == "tax_id"
    assert result["type_confidence"] >= 0.65


def test_classify_utility_bill():
    text = "Electricity Bill\nConsumption: 350 kWh\nService: suministro eléctrico\n"
    result = DocumentIntelligence.analyze(text)
    assert result["document_type"] == "utility_bill"
    assert result["type_confidence"] >= 0.65


def test_classify_bank_statement():
    text = "Bank Statement\nIBAN: DE89370400440532013000\nBalance: 1500.00\n"
    result = DocumentIntelligence.analyze(text)
    assert result["document_type"] == "bank_statement"
    assert result["type_confidence"] >= 0.65


def test_classify_payslip():
    text = "Payslip\nNomina\nSalario: 2500.00\nEmployer: ACME Corp\n"
    result = DocumentIntelligence.analyze(text)
    assert result["document_type"] == "payslip"
    assert result["type_confidence"] >= 0.65


def test_classify_employment_letter():
    text = "Employment Letter\nTo Whom It May Concern\nEmployment at ACME Corp\n"
    result = DocumentIntelligence.analyze(text)
    assert result["document_type"] == "employment_letter"
    assert result["type_confidence"] >= 0.65


def test_classify_generic_document_low_confidence():
    text = "Some random scanned text with no recognizable keywords."
    result = DocumentIntelligence.analyze(text)
    assert result["document_type"] == "generic_document"
    assert result["type_confidence"] < 0.55


# ---------------------------------------------------------------------------
# PersonalDocExtractor tests
# ---------------------------------------------------------------------------


@pytest.fixture
def extractor():
    return PersonalDocExtractor()


def test_extract_passport_fields(extractor):
    text = (
        "PASSPORT\n"
        "Surname: SMITH\n"
        "Given Names: JOHN\n"
        "Nationality: GBR\n"
        "Date of Birth: 15/03/1985\n"
        "Passport No: AB123456\n"
        "Date of Expiry: 25/09/2030\n"
    )
    fields, _ = extractor.extract(text, "passport")
    field_names = {f.name for f in fields}
    assert "date_of_birth" in field_names
    assert "document_number" in field_names
    assert "nationality" in field_names
    assert "expiry_date" in field_names


def test_extract_id_document_fields(extractor):
    text = (
        "IDENTITY CARD\n"
        "DNI: 12345678X\n"
        "JUAN PÉREZ GARCÍA\n"
        "Date of Birth: 12/05/1990\n"
        "Expiry: 01/01/2030\n"
    )
    fields, _ = extractor.extract(text, "id_document")
    field_names = {f.name for f in fields}
    assert "document_number" in field_names
    assert "date_of_birth" in field_names


def test_bank_card_pan_is_masked(extractor):
    text = "VISA\n4111 1111 1111 1111\nJOHN SMITH\nEXP 12/26\n"
    fields, _ = extractor.extract(text, "bank_card")
    card_field = next((f for f in fields if f.name == "card_number"), None)
    assert card_field is not None, "card_number field should be present"
    # Last 4 digits should be visible
    assert "1111" in card_field.value
    # Full PAN must not appear in value or raw_ocr
    assert "4111 1111 1111 1111" not in card_field.value
    assert card_field.raw_ocr == "[REDACTED]"


def test_bank_card_cvv_omitted(extractor):
    """CVV/CVC must never appear in the response fields."""
    text = "4111 1111 1111 1111\nCVV 123\nEXP 12/26\n"
    fields, _ = extractor.extract(text, "bank_card")
    cvv_fields = [f for f in fields if f.name in {"cvv", "cvc", "cvv2", "cvc2"}]
    assert cvv_fields == [], "CVV must be omitted from response"


def test_generic_document_returns_no_fields(extractor):
    fields, warnings = extractor.extract("Some generic text", "generic_document")
    assert fields == []
    assert warnings == []


def test_utility_bill_fields(extractor):
    text = (
        "ELECTRICITY BILL\n"
        "Full Name: JANE DOE\n"
        "Address: 123 Main Street\n"
        "Account Number: ACC-9876543210\n"
        "Pay Period: January 2024\n"
        "Total Amount: $150.00\n"
    )
    fields, _ = extractor.extract(text, "utility_bill")
    field_names = {f.name for f in fields}
    assert "total_amount" in field_names or "period" in field_names


def test_payslip_fields(extractor):
    text = (
        "PAYSLIP\n"
        "Employee: ALICE JONES\n"
        "Employer: ACME Corp\n"
        "Gross Pay: $5,000.00\n"
        "Pay Period: March 2024\n"
    )
    fields, _ = extractor.extract(text, "payslip")
    field_names = {f.name for f in fields}
    assert "employer" in field_names or "salary" in field_names


def test_low_confidence_when_ambiguous_chars(extractor):
    """Fields with '?' markers (uncertain OCR chars) must be 'low' confidence."""
    text = "DNI: 1234?678X\n"
    fields, warnings = extractor.extract(text, "id_document")
    doc_field = next((f for f in fields if f.name == "document_number"), None)
    assert doc_field is not None, (
        "document_number field should be present for DNI input"
    )
    assert doc_field.confidence_level == "low"
    assert any("low confidence" in w for w in warnings)


def test_warnings_generated_for_partial_reconstructions(extractor):
    text = (
        "PASSPORT\n"
        "Date of Birth: 15/03/1985\n"
    )
    fields, warnings = extractor.extract(text, "passport")
    # date_of_birth raw_ocr has '/' normalized to '-', so it triggers warning
    dob_field = next((f for f in fields if f.name == "date_of_birth"), None)
    if dob_field and dob_field.raw_ocr != dob_field.value:
        assert len(warnings) >= 1


def test_detect_metadata_spanish():
    text = "Apellido: GARCIA\nNombre: JUAN\nFecha de nacimiento: 12/05/1990"
    meta = detect_metadata(text)
    assert meta["language_guess"] == "es"


def test_detect_metadata_english_fallback():
    text = "Surname: Smith Given Names: John Date of Birth: 15/03/1985"
    meta = detect_metadata(text)
    assert meta["language_guess"] == "en"


def test_detect_metadata_brazil():
    text = "CPF: 123.456.789-00\nNome: Maria Silva"
    meta = detect_metadata(text)
    assert meta["language_guess"] == "pt"
    assert meta["country_guess"] == "BR"


def test_extract_tax_id_fields(extractor):
    text = "NIF: A1234567B\nFull Name: CARLOS LOPEZ\n"
    fields, _ = extractor.extract(text, "tax_id")
    field_names = {f.name for f in fields}
    assert "tax_number" in field_names


def test_driver_license_fields(extractor):
    text = (
        "DRIVING LICENCE\n"
        "Surname: BROWN\n"
        "Given Names: EMILY\n"
        "Date of Birth: 01/06/1992\n"
        "Expiry: 01/06/2032\n"
        "Address: 42 Oak Lane, London\n"
    )
    fields, _ = extractor.extract(text, "driver_license")
    field_names = {f.name for f in fields}
    assert "date_of_birth" in field_names
    assert "expiry_date" in field_names


# ---------------------------------------------------------------------------
# Luhn algorithm unit tests
# ---------------------------------------------------------------------------


def test_luhn_valid_known_good_card():
    """Standard Visa test PAN 4111111111111111 must pass Luhn."""
    assert _luhn_valid("4111111111111111") is True


def test_luhn_invalid_last_digit_changed():
    """Changing the last digit of a Luhn-valid number must fail."""
    assert _luhn_valid("4111111111111112") is False


def test_luhn_rejects_non_digit_string():
    assert _luhn_valid("4111-1111-1111-1111") is False


def test_luhn_rejects_too_short():
    assert _luhn_valid("123456789012") is False  # 12 digits, below min 13


def test_luhn_rejects_too_long():
    assert _luhn_valid("1" * 20) is False  # 20 digits, above max 19


def test_luhn_amex_valid():
    """Standard Amex test PAN 378282246310005 must pass Luhn."""
    assert _luhn_valid("378282246310005") is True


# ---------------------------------------------------------------------------
# Pattern-aware validator integration tests (Luhn + expiry date)
# ---------------------------------------------------------------------------


def test_luhn_valid_card_boosts_confidence_to_high(extractor):
    """card_number passing Luhn gets confidence_level='high' and a positive note."""
    text = "VISA\n4111 1111 1111 1111\nJOHN SMITH\nEXP 12/26\n"
    fields, warnings = extractor.extract(text, "bank_card")
    card = next((f for f in fields if f.name == "card_number"), None)
    assert card is not None, "card_number field should be extracted"
    assert card.confidence_level == "high"
    assert any("Luhn check passed" in w for w in warnings), warnings


def test_luhn_invalid_card_lowers_confidence_to_low(extractor):
    """card_number failing Luhn gets confidence_level='low' and a warning."""
    text = "VISA\n4111 1111 1111 1112\nJOHN SMITH\nEXP 12/26\n"
    fields, warnings = extractor.extract(text, "bank_card")
    card = next((f for f in fields if f.name == "card_number"), None)
    assert card is not None, "card_number field should be extracted"
    assert card.confidence_level == "low"
    assert any("Luhn check failed" in w for w in warnings), warnings


def test_valid_mmyy_expiry_boosts_confidence_to_high(extractor):
    """card expiry in MM/YY format within valid range → confidence_level='high'."""
    text = "VISA\n4111 1111 1111 1111\nEXP 12/26\n"
    fields, warnings = extractor.extract(text, "bank_card")
    exp = next((f for f in fields if f.name == "expiry_date"), None)
    assert exp is not None, "expiry_date should be extracted"
    assert exp.confidence_level == "high"
    assert any("format valid" in w for w in warnings), warnings


def test_invalid_expiry_month_lowers_confidence(extractor):
    """Expiry with month 13 must get confidence_level='low' and a warning."""
    text = "EXP 13/26\n"
    fields, warnings = extractor.extract(text, "bank_card")
    exp = next((f for f in fields if f.name == "expiry_date"), None)
    if exp:  # only assert if the pattern matched
        assert exp.confidence_level == "low"
        assert any("invalid month" in w for w in warnings), warnings


def test_passport_full_date_expiry_boosts_confidence(extractor):
    """Document expiry in DD/MM/YYYY format → confidence_level='high'."""
    text = "PASSPORT\nDate of Expiry: 25/09/2030\n"
    fields, warnings = extractor.extract(text, "passport")
    exp = next((f for f in fields if f.name == "expiry_date"), None)
    assert exp is not None, "expiry_date should be extracted from passport text"
    assert exp.confidence_level == "high"
    assert any("format valid" in w for w in warnings), warnings


def test_bank_statement_opening_balance_field(extractor):
    """bank_statement must extract opening_balance using dedicated patterns."""
    text = (
        "BANK STATEMENT\n"
        "Account: DE89370400440532013000\n"
        "Opening Balance: £1,200.00\n"
        "Statement Period: January 2024\n"
    )
    fields, _ = extractor.extract(text, "bank_statement")
    field_names = {f.name for f in fields}
    assert "opening_balance" in field_names, (
        "opening_balance must be extracted with _OPENING_BALANCE_PATTERNS"
    )
    ob_field = next(f for f in fields if f.name == "opening_balance")
    assert "1,200" in ob_field.value or "1200" in ob_field.value, (
        f"opening_balance value should contain the extracted amount; "
        f"got {ob_field.value!r}"
    )


