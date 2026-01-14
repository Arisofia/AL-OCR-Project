import pytest
from unittest.mock import MagicMock, patch
from lambda_handler import handler


@pytest.fixture
def s3_event():
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
    with patch("lambda_handler.get_services") as mock_get_services:
        mock_textract = MagicMock()
        mock_storage = MagicMock()
        mock_get_services.return_value = (mock_textract, mock_storage)

        mock_textract.analyze_document.return_value = {"text": "some text"}
        mock_storage.save_json.return_value = True

        response = handler(s3_event, None)

        assert response == {"status": "ok"}
        mock_textract.analyze_document.assert_called_once()
        mock_storage.save_json.assert_called_once()


def test_handler_textract_failure(s3_event):
    """Test handler behavior when Textract fails."""
    with patch("lambda_handler.get_services") as mock_get_services:
        mock_textract = MagicMock()
        mock_storage = MagicMock()
        mock_get_services.return_value = (mock_textract, mock_storage)

        mock_textract.analyze_document.side_effect = Exception("Textract boom")
        mock_storage.save_json.return_value = True

        response = handler(s3_event, None)

        # Handler should report the partial failure and count it
        assert response == {"status": "partial_failure", "failed": 1}
        mock_storage.save_json.assert_called_once()
        args, _ = mock_storage.save_json.call_args
        assert "error" in args[0]
        assert "Textract boom" in args[0]["error"]


def test_handler_missing_info():
    """Test handler with missing bucket or key."""
    bad_event = {"Records": [{"s3": {}}]}
    with patch("lambda_handler.logger") as mock_logger:
        response = handler(bad_event, None)
        assert response == {"status": "ok"}
        mock_logger.warning.assert_called_with("Missing bucket or key in event record")
