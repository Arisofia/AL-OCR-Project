from slowapi.errors import RateLimitExceeded
from ocr_service.utils import limiter as limiter_mod
from ocr_service.main import app


def test_rate_limit_handler_registered():
    """The FastAPI app should register our enhanced rate-limit handler."""
    handler = app.exception_handlers.get(RateLimitExceeded)
    assert handler is limiter_mod._rate_limit_exceeded_handler_with_logging


def test_rate_limit_handler_response_and_logging(monkeypatch):
    """Directly call the handler to verify response payload and structured logging."""
    from fastapi import Request
    import json

    handler = limiter_mod._rate_limit_exceeded_handler_with_logging

    # Create a minimal ASGI scope for the Request
    scope = {"type": "http", "method": "POST", "path": "/presign", "headers": []}
    req = Request(scope)
    # RateLimitExceeded expects a Limit object; construct a lightweight dummy
    # that mimics the string representation used by our handler.
    exc = type("_DummyRL", (), {"__str__": lambda self: "429: 5 per 1 minute"})()

    # Replace the module logger with a mock to assert logging behaviour
    from unittest.mock import MagicMock

    mock = MagicMock()
    monkeypatch.setattr(limiter_mod, "logger", mock)

    resp = handler(req, exc)

    assert resp.status_code == 429
    data = json.loads(resp.body)
    assert data.get("detail") == "Rate limit exceeded"

    # Ensure structured log was emitted with expected payload
    mock.warning.assert_called()
    args, kwargs = mock.warning.call_args
    assert args[0] == "Rate limit exceeded"
    extra = kwargs.get("extra", {})
    assert extra.get("path") == "/presign"
    assert extra.get("detail") == "429: 5 per 1 minute"
