import contextlib
from ocr_service.config import get_settings


def test_hugging_face_token_from_env(monkeypatch):
    """Ensure HUGGING_FACE_HUB_TOKEN is read into Settings via env file or env var."""
    # Clear cached settings
    with contextlib.suppress(AttributeError):
        get_settings.cache_clear()

    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "fake-token-123")
    settings = get_settings()
    assert settings.hugging_face_hub_token == "fake-token-123"


def test_hf_token_absent_returns_none(monkeypatch):
    try:
        get_settings.cache_clear()
    except AttributeError:
        pass

    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    settings = get_settings()
    assert settings.hugging_face_hub_token is None
