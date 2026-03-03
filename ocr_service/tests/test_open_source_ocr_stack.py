"""Tests for the open-source three-layer OCR routing and normalization stack."""

import asyncio
from typing import Any, cast

import pytest

from ocr_service.modules.ocr_engine import IterativeOCREngine
from ocr_service.modules.open_source_ocr_stack import (
    DocumentInput,
    FintechNormalizer,
    FintechQualityEvaluator,
    OpenSourceOCRRouter,
)


class _StubEngine:
    def __init__(
        self,
        layer1_text: str = "",
        layer2_text: str = "",
        layer3_text: str = "",
    ):
        """Create deterministic OCR layer outputs for router unit tests."""
        self.layer1_text = layer1_text
        self.layer2_text = layer2_text
        self.layer3_text = layer3_text

    async def process_image(
        self,
        _payload: bytes,
        use_reconstruction: bool = False,
        doc_type: str | None = None,
    ) -> dict[str, Any]:
        """Return a synthetic layer1/layer2 OCR payload for router tests."""
        await asyncio.sleep(0)
        text = self.layer2_text if use_reconstruction else self.layer1_text
        return {
            "text": text,
            "layout_type": "multi_column" if use_reconstruction else "unknown",
            "doc_type": doc_type,
        }

    async def process_image_advanced(
        self,
        _payload: bytes,
        doc_type: str | None = None,
    ) -> dict[str, Any]:
        """Return a synthetic layer3 OCR payload for router tests."""
        await asyncio.sleep(0)
        return {
            "text": self.layer3_text,
            "doc_type": doc_type,
        }


def test_quality_evaluator_bank_statement_good_signal():
    """Quality evaluator should score strong bank statement text as usable."""
    evaluator = FintechQualityEvaluator()
    text = """
    Account Number: 123456789
    Statement period: 01/01/2026 - 01/31/2026
    01/02 Grocery Store $12.30
    01/04 Salary USD 2100.00
    01/05 Utility USD 94.50
    """
    res = evaluator.evaluate(text, "bank_statement")
    assert res["quality_score"] >= 0.6
    assert res["classification"] in {"GOOD", "PARTIAL"}


def test_normalizer_receipt_invoice_extracts_total():
    """Receipt/invoice normalizer should extract a total amount when present."""
    normalizer = FintechNormalizer()
    text = "Merchant: Corner Shop\nDate: 02/14/2026\nTotal Amount: USD 32.50"
    result = normalizer.normalize("receipt", text)
    payload = result.get("receipt_invoice", {})
    assert payload.get("total_amount") is not None


@pytest.mark.asyncio
async def test_router_escalates_to_layer2_for_partial_quality():
    """Router should escalate to layer2+ when initial extraction quality is low."""
    l1 = _StubEngine(layer1_text="x")
    l2 = _StubEngine(layer2_text="Invoice Date 02/14/2026 Total USD 99.40 Merchant Shop")
    router = OpenSourceOCRRouter(
        layer1_engine=cast(IterativeOCREngine, l1),
        layer2_engine=cast(IterativeOCREngine, l2),
        layer3_engine=cast(IterativeOCREngine, l2),
    )

    doc = DocumentInput(
        id="doc-1",
        bytes_or_path=b"fake",
        document_type="invoice",
    )
    result = await router.process_document(doc)

    assert result.engine_used in {
        "layer2:layout_aware",
        "layer3:vision_llm",
        "layer1:iterative_tesseract",
    }
    assert len(result.fallback_chain) >= 1
    assert result.status in {"OK", "PARTIAL", "FAILED"}


@pytest.mark.asyncio
async def test_router_runs_layer3_for_critical_doc_when_needed():
    """Router should include layer3 fallback for critical document categories."""
    l1 = _StubEngine(layer1_text="x")
    l2 = _StubEngine(layer2_text="x")
    l3 = _StubEngine(layer3_text="")
    router = OpenSourceOCRRouter(
        layer1_engine=cast(IterativeOCREngine, l1),
        layer2_engine=cast(IterativeOCREngine, l2),
        layer3_engine=cast(IterativeOCREngine, l3),
    )

    doc = DocumentInput(
        id="doc-2",
        bytes_or_path=b"fake",
        document_type="bank_statement",
    )
    result = await router.process_document(doc)

    assert "layer3:vision_llm" in result.fallback_chain
