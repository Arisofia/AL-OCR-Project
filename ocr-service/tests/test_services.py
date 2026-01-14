import pytest
from unittest.mock import MagicMock, patch
from services.storage import StorageService
from services.textract import TextractService


@pytest.fixture

def mock_s3_client():
    with patch('boto3.client') as mock_boto:
        mock_s3 = MagicMock()
        mock_boto.return_value = mock_s3
        yield mock_s3

@pytest.fixture
def mock_textract_client():
    with patch('boto3.client') as mock_boto:
        mock_textract = MagicMock()
        mock_boto.return_value = mock_textract
        yield mock_textract

def test_storage_service_upload_file(mock_s3_client):
    service = StorageService(bucket_name="test-bucket")
    content = b"fake-image-content"
    filename = "test.png"

    s3_key = service.upload_file(content, filename, "image/png")

    assert s3_key is not None
    assert "processed/" in s3_key
    assert filename in s3_key
    mock_s3_client.put_object.assert_called_once()

def test_storage_service_save_json(mock_s3_client):
    service = StorageService(bucket_name="test-bucket")
    data = {"key": "value"}
    key = "output.json"

    success = service.save_json(data, key)

    assert success is True
    mock_s3_client.put_object.assert_called_once()

def test_textract_service_analyze_document(mock_textract_client):
    service = TextractService()
    mock_textract_client.analyze_document.return_value = {"Blocks": []}

    result = service.analyze_document("test-bucket", "test-key")

    assert result == {"Blocks": []}
    mock_textract_client.analyze_document.assert_called_once()
