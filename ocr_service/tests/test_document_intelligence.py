"""Tests for document-type detection and card Luhn validation metadata."""

from ocr_service.modules.document_intelligence import DocumentIntelligence


def test_analyze_detects_bank_card_and_valid_luhn():
    text = "Tarjeta Visa 4111 1111 1111 1111 exp 12/26"
    result = DocumentIntelligence.analyze(text)

    assert result["document_type"] == "bank_card"
    assert result["card_analysis"]["detected"] is True
    assert result["card_analysis"]["luhn_valid_count"] == 1
    candidate = result["card_analysis"]["candidates"][0]
    assert candidate["brand_guess"] == "visa"
    assert candidate["masked"].endswith("1111")
    assert "4111 1111 1111 1111" not in candidate["masked"]


def test_analyze_flags_incomplete_card_for_manual_review():
    text = "Card number 4048 3700 045 appears partially visible"
    result = DocumentIntelligence.analyze(text)

    assert result["document_type"] == "bank_card"
    assert result["card_analysis"]["detected"] is True
    assert result["card_analysis"]["luhn_valid_count"] == 0
    assert result["card_analysis"]["requires_manual_review"] is True


def test_analyze_detects_card_without_explicit_keywords():
    text = "4048 3700 0453"
    result = DocumentIntelligence.analyze(text)

    assert result["document_type"] == "bank_card"
    assert result["card_analysis"]["detected"] is True


def test_analyze_detects_invoice_type():
    text = "Factura No. 123\nSubtotal: 10.00\nIVA: 1.20\nTotal: 11.20"
    result = DocumentIntelligence.analyze(text, layout_type="standard_form")

    assert result["document_type"] == "invoice"
    assert result["card_analysis"]["detected"] is False


def test_analyze_uses_layout_fallback_for_statement():
    text = "Movimiento cuenta saldo disponible"
    result = DocumentIntelligence.analyze(text, layout_type="dense_text")

    assert result["document_type"] == "statement"
