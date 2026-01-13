import os
from fastapi.testclient import TestClient
from main import app


def test_reconstruction_enabled(tmp_path):
    # Ensure reconstruction feature is enabled for the test
    os.environ["ENABLE_RECONSTRUCTION"] = "true"
    os.environ["RECON_ITERATIONS"] = "1"

    client = TestClient(app)

    # Use the synthetic pixelated sample generated in ocr-reconstruct tests
    sample = os.path.join(os.path.dirname(__file__), "..", "..", "ocr-reconstruct", "tests", "data", "sample_pixelated.png")
    assert os.path.exists(sample), "Run ocr-reconstruct/tests/generate_samples.py to create sample assets"

    with open(sample, "rb") as fh:
        files = {"file": ("sample_pixelated.png", fh, "image/png")}
        headers = {"X-API-KEY": os.getenv("OCR_API_KEY", "default_secret_key")}
        resp = client.post("/ocr", files=files, headers=headers)

    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert "reconstruction" in data
    assert data["reconstruction"] is not None
    assert "preview_text" in data["reconstruction"]
