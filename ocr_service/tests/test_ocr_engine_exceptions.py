import logging

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
    import cv2

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
    text = await processor.extract_text(img)
    assert text == "textract fallback text"
