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
