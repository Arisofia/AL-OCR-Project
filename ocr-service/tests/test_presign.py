from unittest.mock import patch, MagicMock

import sys

# Mangum isn't available in test env; provide a dummy before importing main
sys.modules.setdefault("mangum", MagicMock(Mangum=MagicMock()))

# Some modules import heavy native libs (pytesseract, cv2). Provide light-weight dummies
sys.modules.setdefault("modules", MagicMock())
sys.modules.setdefault("modules.ocr_engine", MagicMock(IterativeOCREngine=MagicMock()))
sys.modules.setdefault("modules.ocr_config", MagicMock(EngineConfig=MagicMock()))
sys.modules.setdefault("modules.processor", MagicMock(OCRProcessor=MagicMock()))

from fastapi.testclient import TestClient  # noqa: E402

from main import app  # noqa: E402
from config import get_settings  # noqa: E402

client = TestClient(app)


@patch("boto3.client")
def test_generate_presigned_post_success(mock_boto):
    mock_s3 = MagicMock()
    mock_boto.return_value = mock_s3
    mock_s3.generate_presigned_post.return_value = {
        "url": "https://s3.example.com/bucket",
        "fields": {"key": "uploads/test.png", "policy": "abc"},
    }

    settings = get_settings()
    settings.s3_bucket_name = "test-bucket"

    headers = {settings.api_key_header_name: settings.ocr_api_key}
    body = {"key": "uploads/test.png", "content_type": "image/png", "expires_in": 600}

    response = client.post("/presign", json=body, headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert "url" in data and "fields" in data
    assert data["url"] == "https://s3.example.com/bucket"


@patch("boto3.client")
def test_generate_presigned_post_missing_bucket(mock_boto):
    settings = get_settings()
    settings.s3_bucket_name = None

    headers = {settings.api_key_header_name: settings.ocr_api_key}
    body = {"key": "uploads/test.png"}

    response = client.post("/presign", json=body, headers=headers)
    assert response.status_code == 500
    assert response.json()["detail"] == "Infrastructure failure: S3 bucket not configured"
