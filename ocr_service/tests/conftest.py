"""Shared pytest fixtures and test environment defaults for ocr_service tests."""

import os
from unittest.mock import MagicMock, patch

import pytest

os.environ.setdefault("OCR_API_KEY", "test-api-key")
os.environ.setdefault("ENVIRONMENT", "development")
os.environ.setdefault("REDIS_STARTUP_CHECK", "false")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
os.environ.setdefault("OTEL_TRACES_EXPORTER", "none")


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
