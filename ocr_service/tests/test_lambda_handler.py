"""
Test suite for the OCR Lambda handler.
"""

from unittest.mock import patch

import pytest

from ocr_service.lambda_handler import handler


@pytest.fixture
def s3_event():
    """Fixture for a standard S3 event."""
    return {
        "Records": [
            {
                "s3": {
                    "bucket": {"name": "test-bucket"},
                    "object": {"key": "test-file.jpg"},
                }
            }
        ]
    }


def test_handler_success(s3_event):
    """Test successful handler execution."""
    with patch("ocr_service.lambda_handler.worker") as mock_worker:
        response = handler(s3_event, None)
        assert response == {"status": "ok"}
        mock_worker.process_s3_record.assert_called_once()


def test_handler_failure(s3_event):
    """Test handler behavior when processing fails."""
    with patch("ocr_service.lambda_handler.worker") as mock_worker:
        mock_worker.process_s3_record.side_effect = Exception("Boom")
        response = handler(s3_event, None)

        assert response == {"status": "partial_failure", "failed": 1}
        mock_worker.process_s3_record.assert_called_once()


def test_handler_with_aws_request_id(s3_event):
    """Test that handler passes RequestId to worker."""

    class MockContext:
        aws_request_id = "RID-456"

    with patch("ocr_service.lambda_handler.worker") as mock_worker:
        response = handler(s3_event, MockContext())
        assert response == {"status": "ok"}
        mock_worker.process_s3_record.assert_called_once_with(
            s3_event["Records"][0], request_id="RID-456"
        )
