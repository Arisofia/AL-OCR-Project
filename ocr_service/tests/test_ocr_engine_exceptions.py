import logging

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
