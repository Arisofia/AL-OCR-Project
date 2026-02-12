import asyncio
from pathlib import Path

from ocr_service.modules.ocr_engine import IterativeOCREngine


def test_iterative_engine_on_sample_image(monkeypatch):
    # Load a small sample image (from ocr_reconstruct tests)
    sample_path = Path("ocr_reconstruct/tests/data/sample_clean.png")
    image_bytes = sample_path.read_bytes()

    # Mock reconstruction to return enhanced bytes and metadata
    def fake_recon_process_bytes(*_args, **_kwargs):
        return ("recon_text", image_bytes, {"meta": "ok"})

    monkeypatch.setattr(
        "ocr_service.modules.ocr_engine.recon_process_bytes",
        fake_recon_process_bytes,
    )

    # Mock pytesseract to return predictable text
    def fake_image_to_string(*_args, **_kwargs):
        return "recognized text"

    monkeypatch.setattr("pytesseract.image_to_string", fake_image_to_string)

    engine = IterativeOCREngine()

    res = asyncio.run(engine.process_image(image_bytes, use_reconstruction=True))

    assert "error" not in res
    assert res.get("success") is True or res.get("text") is not None
    # reconstruction info should be present from our fake recon
    assert res.get("reconstruction") is not None
