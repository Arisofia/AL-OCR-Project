import os
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from main import app

def test_reconstruction_enabled(tmp_path):
    # Clear lru_cache for settings to ensure environment variables are picked up
    from config import get_settings
    get_settings.cache_clear()
    
    # Ensure reconstruction feature is enabled for the test
    os.environ["ENABLE_RECONSTRUCTION"] = "true"
    os.environ["OCR_ITERATIONS"] = "1"

    client = TestClient(app)

    # Use the synthetic pixelated sample generated in ocr-reconstruct tests
    sample = os.path.join(os.path.dirname(__file__), "..", "..", "ocr-reconstruct", "tests", "data", "sample_pixelated.png")
    
    if not os.path.exists(sample):
        # Create a dummy image for the test if the sample is missing
        import numpy as np
        import cv2
        dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.putText(dummy_img, "RECON TEST", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        sample = os.path.join(tmp_path, "dummy_pixelated.png")
        cv2.imwrite(str(sample), dummy_img)

    with open(sample, "rb") as fh:
        files = {"file": ("sample_pixelated.png", fh, "image/png")}
        headers = {"X-API-KEY": os.getenv("OCR_API_KEY", "default_secret_key")}
        
        # Mock the engine's run_reconstruction to return something
        with patch("modules.ocr_engine.IterativeOCREngine._run_reconstruction") as mock_recon:
            mock_recon.return_value = ({"preview_text": "recon text", "meta": "data"}, None)
            resp = client.post("/ocr", files=files, headers=headers)

    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert "reconstruction" in data
    assert data["reconstruction"] is not None
    assert "preview_text" in data["reconstruction"]
