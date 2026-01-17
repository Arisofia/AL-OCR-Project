from unittest.mock import patch

import pytest
from botocore.exceptions import ClientError  # type: ignore

from ocr_service.services.storage import StorageService
from ocr_service.services.textract import TextractService


def test_storage_put_object_retry_success(mock_s3_client):
    """Test S3 put_object success after retries."""
    service = StorageService(bucket_name="test-bucket")

    # Fail once, then succeed
    mock_s3_client.put_object.side_effect = [
        ClientError(
            {"Error": {"Code": "Throttling", "Message": "Rate exceeded"}},
            "PutObject",
        ),
        {"ResponseMetadata": {"HTTPStatusCode": 200}},
    ]

    with patch("time.sleep", return_value=None):  # Skip actual sleeping
        success = service.put_object("key", b"body", "text/plain")

    assert success is True
    assert mock_s3_client.put_object.call_count == 2


def test_storage_put_object_exhaust_retries(mock_s3_client):
    """Test S3 put_object failure after exhausting retries."""
    service = StorageService(bucket_name="test-bucket")
    service.max_retries = 2

    mock_s3_client.put_object.side_effect = ClientError(
        {"Error": {"Code": "InternalError", "Message": "Server Error"}}, "PutObject"
    )

    with patch("time.sleep", return_value=None):
        success = service.put_object("key", b"body", "text/plain")

    assert success is False
    assert mock_s3_client.put_object.call_count == 2


def test_textract_analyze_retry_success(mock_textract_client):
    """Test Textract analysis success after retries."""
    service = TextractService()
    service.max_retries = 3

    mock_textract_client.analyze_document.side_effect = [
        ClientError(
            {
                "Error": {
                    "Code": "ProvisionedThroughputExceededException",
                    "Message": "Slow down",
                }
            },
            "AnalyzeDocument",
        ),
        {"Blocks": []},
    ]

    with patch("time.sleep", return_value=None):
        result = service.analyze_document("bucket", "key")

    assert result == {"Blocks": []}
    assert mock_textract_client.analyze_document.call_count == 2


def test_textract_analyze_persistent_failure(mock_textract_client):
    """Test Textract analysis persistent failure."""
    service = TextractService()
    service.max_retries = 2

    mock_textract_client.analyze_document.side_effect = ClientError(
        {"Error": {"Code": "InternalServerError", "Message": "Oops"}}, "AnalyzeDocument"
    )

    with (
        patch("time.sleep", return_value=None),
        pytest.raises(RuntimeError, match="Max retry threshold reached"),
    ):
        service.analyze_document("bucket", "key")

    assert mock_textract_client.analyze_document.call_count == 2
