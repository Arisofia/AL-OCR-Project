"""Regression tests for OCR engine fallbacks and preprocessing behavior."""

# OpenCV binary modules commonly trigger false `no-member` in pylint.
# pylint: disable=missing-function-docstring,no-member,import-error

import logging

import cv2
import numpy as np
import pytest

import ocr_service.modules.ocr_engine as engine_mod
from ocr_service.modules.ocr_engine import DocumentContext, DocumentProcessor


@pytest.mark.asyncio
async def test_reconstruction_logs_exception(caplog, monkeypatch):
    caplog.set_level(logging.ERROR)

    # Replace the recon_process_bytes with a sync function that raises
    def bad_sync(*_args, **_kwargs):
        raise RuntimeError("recon-failed")

    monkeypatch.setattr(engine_mod, "recon_process_bytes", bad_sync)

    monkeypatch.setattr(engine_mod, "recon_process_bytes", bad_sync)

    processor = DocumentProcessor(
        enhancer=engine_mod.ImageEnhancer(),
        ocr_config=engine_mod.TesseractConfig(),
        engine_config=engine_mod.EngineConfig(),
        reconstructor=None,
    )

    ctx = DocumentContext(
        image_bytes=b"abc", use_reconstruction=True, doc_type="invoice"
    )

    await processor.run_reconstruction(ctx, max_iterations=1)

    assert any(
        "Reconstruction pipeline failed" in r.getMessage() for r in caplog.records
    )


@pytest.mark.asyncio
async def test_document_processor_applies_upscaling():
    # Create a small test image
    small_img = np.zeros((200, 200, 3), dtype=np.uint8)

    _, img_bytes = cv2.imencode(".png", small_img)

    config = engine_mod.EngineConfig(max_upscale_factor=2.0, max_long_side_px=600)
    processor = DocumentProcessor(
        enhancer=engine_mod.ImageEnhancer(),
        ocr_config=engine_mod.TesseractConfig(),
        engine_config=config,
        reconstructor=None,
    )

    ctx = DocumentContext(
        image_bytes=img_bytes.tobytes(), use_reconstruction=False, doc_type="generic"
    )

    success = await processor.decode_and_validate(ctx)
    assert success
    assert ctx.current_img.shape[0] > 200  # Should be upscaled


@pytest.mark.asyncio
async def test_decode_and_validate_handles_unexpected_preprocess_error(monkeypatch):
    """Unexpected preprocessing errors should return False, not raise."""
    processor = DocumentProcessor(
        enhancer=engine_mod.ImageEnhancer(),
        ocr_config=engine_mod.TesseractConfig(),
        engine_config=engine_mod.EngineConfig(max_upscale_factor=2.0),
        reconstructor=None,
    )
    ctx = DocumentContext(
        image_bytes=b"valid-bytes", use_reconstruction=False, doc_type="generic"
    )

    async def _decode_ok(_bytes):
        return np.zeros((10, 10, 3), dtype=np.uint8)

    def _upscale_fail(*_args, **_kwargs):
        raise RuntimeError("simulated resize failure")

    monkeypatch.setattr(engine_mod.ImageToolkit, "decode_image_async", _decode_ok)
    monkeypatch.setattr(engine_mod.ImageToolkit, "upscale_for_ocr", _upscale_fail)

    success = await processor.decode_and_validate(ctx)
    assert success is False


@pytest.mark.asyncio
async def test_process_image_uses_direct_fallback_when_decode_fails(monkeypatch):
    """Engine should use Pillow direct fallback if OpenCV decode path is unavailable."""

    async def _decode_fail(_self, _ctx):
        return False

    async def _direct_text(_self, _image_bytes):
        return "fallback OCR text"

    monkeypatch.setattr(DocumentProcessor, "decode_and_validate", _decode_fail)
    monkeypatch.setattr(DocumentProcessor, "extract_text_direct", _direct_text)

    engine = engine_mod.IterativeOCREngine()
    result = await engine.process_image(b"not-empty")

    assert result.get("success") is True
    assert result.get("method") == "pillow-direct-fallback"
    assert result.get("text") == "fallback OCR text"


@pytest.mark.asyncio
async def test_extract_text_falls_back_to_textract_when_tesseract_missing(monkeypatch):
    processor = DocumentProcessor(
        enhancer=engine_mod.ImageEnhancer(),
        ocr_config=engine_mod.TesseractConfig(),
        engine_config=engine_mod.EngineConfig(),
        reconstructor=None,
    )

    def _raise_tesseract_missing(*_args, **_kwargs):
        raise engine_mod.pytesseract.pytesseract.TesseractNotFoundError()

    async def _fake_textract(_self, _image_bytes):
        return "textract fallback text"

    monkeypatch.setattr(
        engine_mod.pytesseract, "image_to_string", _raise_tesseract_missing
    )
    monkeypatch.setattr(DocumentProcessor, "extract_text_textract", _fake_textract)

    img = np.zeros((16, 16, 3), dtype=np.uint8)
    original_bytes = b"fake image bytes"
    text = await processor.extract_text(img, original_bytes=original_bytes)
    assert text == "textract fallback text"


@pytest.mark.asyncio
async def test_extract_text_falls_back_to_textract_on_tesseract_runtime_error(
    monkeypatch,
):
    processor = DocumentProcessor(
        enhancer=engine_mod.ImageEnhancer(),
        ocr_config=engine_mod.TesseractConfig(),
        engine_config=engine_mod.EngineConfig(),
        reconstructor=None,
    )

    def _raise_tesseract_runtime(*_args, **_kwargs):
        raise engine_mod.pytesseract.pytesseract.TesseractError(1, "runtime failure")

    async def _fake_textract(_self, _image_bytes):
        return "textract runtime fallback"

    monkeypatch.setattr(
        engine_mod.pytesseract, "image_to_string", _raise_tesseract_runtime
    )
    monkeypatch.setattr(DocumentProcessor, "extract_text_textract", _fake_textract)

    img = np.zeros((16, 16, 3), dtype=np.uint8)
    original_bytes = b"fake image bytes"
    text = await processor.extract_text(img, original_bytes=original_bytes)
    assert text == "textract runtime fallback"


@pytest.mark.asyncio
async def test_process_image_applies_textract_quality_fallback(monkeypatch):
    engine = engine_mod.IterativeOCREngine(
        config=engine_mod.EngineConfig(max_iterations=1, confidence_threshold=0.5)
    )
    calls = {"textract": 0, "direct": 0}

    async def _decode_ok(ctx):
        ctx.current_img = np.zeros((32, 32, 3), dtype=np.uint8)
        return True

    async def _recon_noop(_ctx, _max_iterations):
        return None

    async def _layout_noop(ctx):
        ctx.layout_regions = []
        ctx.layout_type = "unknown"

    def _preprocess(_img, _iteration, _use_recon):
        return np.zeros((32, 32), dtype=np.uint8)

    async def _extract_low_quality(_img, _regions=None, _original_bytes=None):
        return "IR {g W rm"

    async def _textract_good(_image_bytes):
        calls["textract"] += 1
        return ("Factura total fecha nombre id " * 8).strip()

    async def _direct_empty(_image_bytes):
        calls["direct"] += 1
        return ""

    async def _enhance_noop(img):
        return img

    monkeypatch.setattr(engine.processor, "decode_and_validate", _decode_ok)
    monkeypatch.setattr(engine.processor, "run_reconstruction", _recon_noop)
    monkeypatch.setattr(engine, "_analyze_layout", _layout_noop)
    monkeypatch.setattr(engine.processor, "preprocess_frame", _preprocess)
    monkeypatch.setattr(engine.processor, "extract_text", _extract_low_quality)
    monkeypatch.setattr(engine.processor, "extract_text_textract", _textract_good)
    monkeypatch.setattr(engine.processor, "extract_text_direct", _direct_empty)
    monkeypatch.setattr(engine_mod.ImageToolkit, "enhance_iteration", _enhance_noop)

    result = await engine.process_image(b"img-bytes")

    assert calls["textract"] == 1
    assert calls["direct"] == 1
    assert "Factura total" in result.get("text", "")
    assert any(
        i.get("method") == "textract-quality-fallback"
        for i in result.get("iterations", [])
    )


@pytest.mark.asyncio
async def test_process_image_skips_textract_fallback_on_high_confidence(monkeypatch):
    engine = engine_mod.IterativeOCREngine(
        config=engine_mod.EngineConfig(max_iterations=1, confidence_threshold=0.5)
    )
    calls = {"textract": 0}

    async def _decode_ok(ctx):
        ctx.current_img = np.zeros((32, 32, 3), dtype=np.uint8)
        return True

    async def _recon_noop(_ctx, _max_iterations):
        return None

    async def _layout_noop(ctx):
        ctx.layout_regions = []
        ctx.layout_type = "unknown"

    def _preprocess(_img, _iteration, _use_recon):
        return np.zeros((32, 32), dtype=np.uint8)

    async def _extract_high_quality(_img, _regions=None, _original_bytes=None):
        return ("Factura total fecha nombre id " * 8).strip()

    async def _textract_unused(_image_bytes):
        calls["textract"] += 1
        return "textract should not be used"

    async def _enhance_noop(img):
        return img

    monkeypatch.setattr(engine.processor, "decode_and_validate", _decode_ok)
    monkeypatch.setattr(engine.processor, "run_reconstruction", _recon_noop)
    monkeypatch.setattr(engine, "_analyze_layout", _layout_noop)
    monkeypatch.setattr(engine.processor, "preprocess_frame", _preprocess)
    monkeypatch.setattr(engine.processor, "extract_text", _extract_high_quality)
    monkeypatch.setattr(engine.processor, "extract_text_textract", _textract_unused)
    monkeypatch.setattr(engine_mod.ImageToolkit, "enhance_iteration", _enhance_noop)

    result = await engine.process_image(b"img-bytes")

    assert calls["textract"] == 0
    assert "Factura total" in result.get("text", "")
    assert not any(
        i.get("method") == "textract-quality-fallback"
        for i in result.get("iterations", [])
    )


@pytest.mark.asyncio
async def test_process_image_applies_direct_quality_fallback(monkeypatch):
    engine = engine_mod.IterativeOCREngine(
        config=engine_mod.EngineConfig(max_iterations=1, confidence_threshold=0.5)
    )
    calls = {"textract": 0, "direct": 0}

    async def _decode_ok(ctx):
        ctx.current_img = np.zeros((32, 32, 3), dtype=np.uint8)
        return True

    async def _recon_noop(_ctx, _max_iterations):
        return None

    async def _layout_noop(ctx):
        ctx.layout_regions = []
        ctx.layout_type = "unknown"

    def _preprocess(_img, _iteration, _use_recon):
        return np.zeros((32, 32), dtype=np.uint8)

    async def _extract_low_quality(_img, _regions=None, _original_bytes=None):
        return "IR {g W rm"

    async def _textract_weak(_image_bytes):
        calls["textract"] += 1
        return "IR {g W rm"

    async def _direct_good(_image_bytes):
        calls["direct"] += 1
        return ("Factura total fecha nombre id " * 8).strip()

    async def _enhance_noop(img):
        return img

    monkeypatch.setattr(engine.processor, "decode_and_validate", _decode_ok)
    monkeypatch.setattr(engine.processor, "run_reconstruction", _recon_noop)
    monkeypatch.setattr(engine, "_analyze_layout", _layout_noop)
    monkeypatch.setattr(engine.processor, "preprocess_frame", _preprocess)
    monkeypatch.setattr(engine.processor, "extract_text", _extract_low_quality)
    monkeypatch.setattr(engine.processor, "extract_text_textract", _textract_weak)
    monkeypatch.setattr(engine.processor, "extract_text_direct", _direct_good)
    monkeypatch.setattr(engine_mod.ImageToolkit, "enhance_iteration", _enhance_noop)

    result = await engine.process_image(b"img-bytes")

    assert calls["textract"] == 1
    assert calls["direct"] == 1
    assert "Factura total" in result.get("text", "")
    assert any(
        i.get("method") == "direct-quality-fallback"
        for i in result.get("iterations", [])
    )


def test_sanitize_text_normalizes_grouped_numeric_noise():
    processor = DocumentProcessor(
        enhancer=engine_mod.ImageEnhancer(),
        ocr_config=engine_mod.TesseractConfig(),
        engine_config=engine_mod.EngineConfig(),
        reconstructor=None,
    )

    # Noise punctuation around grouped digits should be normalized.
    cleaned = processor.sanitize_text("4048-.. 3700 045——")
    assert cleaned == "4048 3700 045"

    cleaned_single = processor.sanitize_text("4048-. 3700 045—")
    assert cleaned_single == "4048 3700 045"

    # Keep decimal/monetary formatting readable.
    amount = processor.sanitize_text("Total: 1.250,00 EUR")
    assert "1.250,00" in amount


def test_preprocess_frame_uses_clean_for_ocr():
    """Verify that the cleaning pipeline is used in the first iteration."""
    processor = DocumentProcessor(
        enhancer=engine_mod.ImageEnhancer(),
        ocr_config=engine_mod.TesseractConfig(),
        engine_config=engine_mod.EngineConfig(),
        reconstructor=None,
    )

    # Create a noisy image
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

    # First iteration (0) should use clean_for_ocr
    thresh = processor.preprocess_frame(img, iteration=0, use_recon=False)

    # clean_for_ocr returns a binary image (thresholded)
    assert len(thresh.shape) == 2
    assert thresh.dtype == np.uint8
    assert set(np.unique(thresh)).issubset({0, 255})


def test_preprocess_frame_later_iterations_apply_pixel_rescue():
    """Later iterations should upscale/denoise to rescue faint pixel detail."""
    processor = DocumentProcessor(
        enhancer=engine_mod.ImageEnhancer(),
        ocr_config=engine_mod.TesseractConfig(),
        engine_config=engine_mod.EngineConfig(),
        reconstructor=None,
    )

    img = np.random.randint(0, 255, (80, 120, 3), dtype=np.uint8)
    rescued = processor.preprocess_frame(img, iteration=2, use_recon=False)

    assert rescued.shape[0] > img.shape[0]
    assert rescued.shape[1] > img.shape[1]
    assert rescued.dtype == np.uint8


@pytest.mark.asyncio
async def test_extract_text_applies_digit_rescue_on_ambiguous_output(monkeypatch):
    processor = DocumentProcessor(
        enhancer=engine_mod.ImageEnhancer(),
        ocr_config=engine_mod.TesseractConfig(),
        engine_config=engine_mod.EngineConfig(),
        reconstructor=None,
    )
    calls = []

    def _fake_ocr(_img, config):
        calls.append(config)
        if "tessedit_char_whitelist=0123456789" in config:
            return "4048 3700 0453"
        return "4048 3700 04M!"

    monkeypatch.setattr(engine_mod.pytesseract, "image_to_string", _fake_ocr)
    img = np.zeros((80, 240, 3), dtype=np.uint8)

    result = await processor.extract_text(img)

    assert result == "4048 3700 0453"
    assert any("tessedit_char_whitelist=0123456789" in c for c in calls)


def test_digit_rescue_not_required_for_clean_numeric_text():
    processor = DocumentProcessor(
        enhancer=engine_mod.ImageEnhancer(),
        ocr_config=engine_mod.TesseractConfig(),
        engine_config=engine_mod.EngineConfig(),
        reconstructor=None,
    )

    assert processor._needs_digit_rescue("4048 3700 0453") is False
    assert processor._needs_digit_rescue("4048 3700 04M!") is True
