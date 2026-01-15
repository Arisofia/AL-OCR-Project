import os
from unittest.mock import patch

from fastapi.testclient import TestClient
from main import app


def _get_test_image(base_path, tmp_path):
    if os.path.exists(base_path):
        return base_path

    import cv2
    import numpy as np

    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.putText(
        dummy_img,
        "RECON TEST",
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )
    sample = os.path.join(tmp_path, "dummy_pixelated.png")
    cv2.imwrite(str(sample), dummy_img)
    return sample


def test_reconstruction_enabled(tmp_path, monkeypatch):
    # Clear lru_cache for settings to ensure environment variables are picked up
    from config import get_settings

    get_settings.cache_clear()

    # Ensure reconstruction is enabled for this test (monkeypatch used)
    test_key = "test_recon_key"
    monkeypatch.setenv("ENABLE_RECONSTRUCTION", "true")
    monkeypatch.setenv("OCR_ITERATIONS", "1")
    monkeypatch.setenv("OCR_API_KEY", test_key)

    client = TestClient(app)

    # Use the synthetic pixelated sample generated in ocr_reconstruct tests
    base_dir = os.path.dirname(__file__)
    sample = os.path.join(
        base_dir,
        "..",
        "..",
        "ocr_reconstruct",
        "tests",
        "data",
        "sample_pixelated.png",
    )
    sample = _get_test_image(sample, tmp_path)

    with open(sample, "rb") as fh:
        files = {
            "file": ("sample_pixelated.png", fh, "image/png"),
        }
        headers = {"X-API-KEY": test_key}

        # Mock the engine's run_reconstruction to return something
        with patch(
            "modules.ocr_engine.IterativeOCREngine._run_reconstruction"
        ) as mock_recon:
            mock_recon.return_value = (
                {"preview_text": "recon text", "meta": "data"},
                None,
            )
            resp = client.post("/ocr", files=files, headers=headers)

    assert resp.status_code == 200
    data = resp.json()
    assert "reconstruction" in data
    assert data["reconstruction"] is not None
    assert "preview_text" in data["reconstruction"]
