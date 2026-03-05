"""Tests for new field patterns and document type aliases in personal_doc_extractor."""

import pytest

from ocr_service.modules.personal_doc_extractor import (
    FIELD_DEFINITIONS,
    PersonalDocExtractor,
)


@pytest.fixture
def extractor():
    return PersonalDocExtractor()


def test_id_card_alias_extraction(extractor):
    """'id_card' type must use same field extraction as 'national_id'."""
    text = (
        "IDENTITY CARD\n"
        "DNI: 12345678X\n"
        "JUAN PEREZ\n"
        "Date of Birth: 12/05/1990\n"
        "Expiry: 01/01/2030\n"
    )
    fields, _ = extractor.extract(text, "id_card")
    field_names = {f.name for f in fields}
    assert "document_number" in field_names
    assert "date_of_birth" in field_names


def test_credit_card_alias(extractor):
    """'credit_card' type must extract card_number and expiry_date."""
    text = "VISA\n4111 1111 1111 1111\nJOHN SMITH\nEXP 12/26\n"
    fields, _ = extractor.extract(text, "credit_card")
    field_names = {f.name for f in fields}
    assert "card_number" in field_names
    assert "expiry_date" in field_names


def test_debit_card_alias(extractor):
    """'debit_card' type must extract card_number and expiry_date."""
    text = "MASTERCARD\n5500 0000 0000 0004\nALICE JONES\nEXP 11/27\n"
    fields, _ = extractor.extract(text, "debit_card")
    field_names = {f.name for f in fields}
    assert "card_number" in field_names
    assert "expiry_date" in field_names


def test_invoice_vat_extraction(extractor):
    text = (
        "INVOICE\n"
        "Total Amount: €500.00\n"
        "VAT: €95.00\n"
        "Issue Date: 15/01/2024\n"
    )
    fields, _ = extractor.extract(text, "invoice")
    field_names = {f.name for f in fields}
    assert "vat_amount" in field_names
    vat_field = next(f for f in fields if f.name == "vat_amount")
    assert "95" in vat_field.value


def test_invoice_issue_date_extraction(extractor):
    text = (
        "INVOICE\n"
        "Issue Date: 15/01/2024\n"
        "Total Amount: €500.00\n"
    )
    fields, _ = extractor.extract(text, "invoice")
    field_names = {f.name for f in fields}
    assert "issue_date" in field_names


def test_bank_statement_closing_balance(extractor):
    text = (
        "BANK STATEMENT\n"
        "Account: DE89370400440532013000\n"
        "Opening Balance: £1,200.00\n"
        "Closing Balance: $5,432.10\n"
        "Statement Period: January 2024\n"
    )
    fields, _ = extractor.extract(text, "bank_statement")
    field_names = {f.name for f in fields}
    assert "closing_balance" in field_names
    cb_field = next(f for f in fields if f.name == "closing_balance")
    assert "5,432" in cb_field.value or "5432" in cb_field.value


def test_passport_place_of_birth(extractor):
    text = (
        "PASSPORT\n"
        "Surname: GARCIA\n"
        "Given Names: MARIA\n"
        "Place of Birth: MADRID\n"
        "Date of Birth: 01/01/1990\n"
        "Passport No: A1234567\n"
        "Expiry: 01/01/2030\n"
    )
    fields, _ = extractor.extract(text, "passport")
    field_names = {f.name for f in fields}
    assert "place_of_birth" in field_names
    pob_field = next(f for f in fields if f.name == "place_of_birth")
    assert "MADRID" in pob_field.value.upper()


def test_id_card_place_of_birth(extractor):
    text = (
        "IDENTITY CARD\n"
        "DNI: 12345678X\n"
        "Full Name: CARLOS LOPEZ\n"
        "Place of Birth: BARCELONA\n"
        "Date of Birth: 15/03/1985\n"
    )
    fields, _ = extractor.extract(text, "id_card")
    field_names = {f.name for f in fields}
    assert "place_of_birth" in field_names


def test_id_card_in_field_definitions():
    assert "id_card" in FIELD_DEFINITIONS


def test_credit_card_in_field_definitions():
    assert "credit_card" in FIELD_DEFINITIONS


def test_debit_card_in_field_definitions():
    assert "debit_card" in FIELD_DEFINITIONS


def test_bank_statement_has_closing_balance():
    field_names = [fd[0] for fd in FIELD_DEFINITIONS["bank_statement"]]
    assert "closing_balance" in field_names


def test_invoice_has_vat_amount():
    field_names = [fd[0] for fd in FIELD_DEFINITIONS["invoice"]]
    assert "vat_amount" in field_names


def test_invoice_has_issue_date():
    field_names = [fd[0] for fd in FIELD_DEFINITIONS["invoice"]]
    assert "issue_date" in field_names


def test_passport_has_place_of_birth():
    field_names = [fd[0] for fd in FIELD_DEFINITIONS["passport"]]
    assert "place_of_birth" in field_names
