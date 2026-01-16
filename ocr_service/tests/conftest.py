import os
from unittest.mock import MagicMock, patch

import pytest

# Set dummy environment variables for Pydantic Settings validation
os.environ.setdefault("OCR_API_KEY", "test-api-key")
os.environ.setdefault("ENVIRONMENT", "development")


@pytest.fixture
def mock_s3_client():
    """Mock S3 client fixture."""
    with patch("boto3.client") as mock_boto:
        mock_s3 = MagicMock()
        mock_boto.return_value = mock_s3
        yield mock_s3


@pytest.fixture
def mock_textract_client():
    """Mock Textract client fixture."""
    with patch("boto3.client") as mock_boto:
        mock_textract = MagicMock()
        mock_boto.return_value = mock_textract
        yield mock_textract
