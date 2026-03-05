"""Tests for OCR response document-type behavior."""

from typing import Any, cast

from ocr_service.modules.document_intelligence import DocumentIntelligence
from ocr_service.modules.ocr_engine import DocumentContext, IterativeOCREngine


class _FakeProcessor:
    def sanitize_text(self, text: str) -> str:
        return text

    def mark_uncertain_partial_card_tail(self, text: str) -> str:
        return text


class _FakeConfig:
    enable_bin_lookup = False


def test_build_response_prefers_requested_doc_type_when_text_is_empty(monkeypatch):
    """Explicit doc_type should be kept when classifier fallback is weak/empty."""
    engine = IterativeOCREngine.__new__(IterativeOCREngine)
    engine_for_test = cast(Any, engine)
    engine_for_test.processor = _FakeProcessor()
    engine_for_test.config = _FakeConfig()

    monkeypatch.setattr(
        DocumentIntelligence,
        "analyze",
        lambda *_args, **_kwargs: {
            "document_type": "statement",
            "type_confidence": 0.60,
        },
    )

    ctx = DocumentContext(
        image_bytes=b"img",
        use_reconstruction=False,
        doc_type="bank_card",
        layout_type="dense_text",
        best_text="",
        best_confidence=0.0,
    )

    resp = engine._build_response(ctx)

    assert resp["document_type"] == "bank_card"
    assert resp["type_confidence"] >= 0.50
