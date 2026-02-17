"""Tests for document-type detection and card Luhn validation metadata."""

import httpx

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


def test_fetch_bin_info_returns_metadata(monkeypatch):
    class _Resp:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "bank": {"name": "Diners Club Ecuador"},
                "country": {"name": "Ecuador"},
                "type": "credit",
                "brand": "Diners Club",
                "scheme": "diners",
            }

    def _fake_get(url, headers=None, timeout=None):
        assert "lookup.binlist.net/40483700" in url
        assert headers == {"Accept-Version": "3"}
        assert timeout is not None
        return _Resp()

    DocumentIntelligence._BIN_INFO_CACHE.clear()
    monkeypatch.setattr(httpx, "get", _fake_get)

    result = DocumentIntelligence.fetch_bin_info("40483700")

    assert result is not None
    assert result["issuer"] == "Diners Club Ecuador"
    assert result["country"] == "Ecuador"
    assert result["type"] == "credit"


def test_analyze_includes_bin_info_when_enabled(monkeypatch):
    def _fake_bin_lookup(_prefix):
        return {
            "issuer": "Diners Club Ecuador",
            "country": "Ecuador",
            "type": "credit",
            "brand": "Diners Club",
            "scheme": "diners",
        }

    monkeypatch.setattr(DocumentIntelligence, "fetch_bin_info", _fake_bin_lookup)

    result = DocumentIntelligence.analyze(
        "Card 4048 3700 0453 0003",
        include_bin_info=True,
    )

    candidate = result["card_analysis"]["candidates"][0]
    assert "bin_info" in candidate
    assert candidate["bin_info"]["issuer"] == "Diners Club Ecuador"
