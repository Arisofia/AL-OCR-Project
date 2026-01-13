import pytest
import json
from unittest.mock import MagicMock, patch
from lambda_handler import handler

@pytest.fixture
def s3_event():
    return {
        "Records": [
            {
                "s3": {
                    "bucket": {"name": "test-bucket"},
                    "object": {"key": "test-file.jpg"}
                }
            }
        ]
    }

def test_handler_success(s3_event):
    with patch('lambda_handler.get_services') as mock_get_services:
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
    with patch('lambda_handler.get_services') as mock_get_services:
        mock_textract = MagicMock()
        mock_storage = MagicMock()
        mock_get_services.return_value = (mock_textract, mock_storage)
        
        mock_textract.analyze_document.side_effect = Exception("Textract boom")
        mock_storage.save_json.return_value = True
        
        response = handler(s3_event, None)
        
        assert response == {"status": "ok"} # Handler catches individual record failures
        # Should have called save_json twice (one for output, one for error? No, just once for error in this case)
        # Wait, if analyze_document fails, it goes to except block and calls save_json with err_obj
        mock_storage.save_json.assert_called_once()
        args, _ = mock_storage.save_json.call_args
        assert "error" in args[0]
        assert "Textract boom" in args[0]["error"]

def test_handler_missing_info():
    bad_event = {"Records": [{"s3": {}}]}
    with patch('lambda_handler.logger') as mock_logger:
        response = handler(bad_event, None)
        assert response == {"status": "ok"}
        mock_logger.warning.assert_called_with('Missing bucket or key in event record')
