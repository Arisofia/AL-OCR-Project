import pytest
from pydantic import ValidationError

from ocr_service.config import Settings


def test_allowed_origins_wildcard_development():
    """Wildcard origins are allowed in development."""
    settings = Settings(
        ocr_api_key="test", environment="development", allowed_origins=["*"]
    )
    assert settings.allowed_origins == ["*"]


def test_allowed_origins_wildcard_production():
    """Wildcard origins are NOT allowed in production."""
    with pytest.raises(ValidationError) as excinfo:
        Settings(ocr_api_key="test", environment="production", allowed_origins=["*"])
    assert "Wildcard CORS origins are not allowed in production." in str(excinfo.value)


def test_allowed_origins_specific_production():
    """Specific origins are allowed in production."""
    settings = Settings(
        ocr_api_key="test",
        environment="production",
        allowed_origins=["https://example.com"],
    )
    assert settings.allowed_origins == ["https://example.com"]
