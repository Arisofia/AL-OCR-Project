from slowapi.errors import RateLimitExceeded

from ocr_service.main import app
from ocr_service.utils import limiter as limiter_mod


def test_rate_limit_handler_registered():
    """The FastAPI app should register our enhanced rate-limit handler."""
    handler = app.exception_handlers.get(RateLimitExceeded)
    assert handler is limiter_mod._rate_limit_exceeded_handler_with_logging


def test_rate_limit_handler_response_and_logging(monkeypatch):
    """Directly call the handler to verify response payload and structured logging."""
    import json

    from fastapi import Request

    handler = limiter_mod._rate_limit_exceeded_handler_with_logging

    # Create a minimal ASGI scope for the Request
    scope = {"type": "http", "method": "POST", "path": "/presign", "headers": []}
    req = Request(scope)

    # Use a tiny dummy instead of constructing a full Limit-backed exception
    def _dummy_str(_self):
        return "429: 5 per 1 minute"

    exc = type(
        "_DummyRL",
        (),
        {"__str__": _dummy_str},
    )()

    # Replace the module logger with a mock to assert logging behaviour
    from unittest.mock import MagicMock

    mock = MagicMock()
    monkeypatch.setattr(limiter_mod, "logger", mock)

    resp = handler(req, exc)  # type: ignore

    assert resp.status_code == 429
    data = json.loads(resp.body)
    assert data.get("detail") == "Rate limit exceeded. Please try again later."

    # Ensure structured log was emitted with expected payload
    mock.warning.assert_called()
    _, kwargs = mock.warning.call_args
    assert kwargs.get("extra", {}) != {}
    extra = kwargs.get("extra", {})
    assert extra.get("path") == "/presign"
    assert extra.get("detail") == "429: 5 per 1 minute"


def test_rate_limit_handler_response_shape(monkeypatch):
    """The handler should return a 429 JSONResponse with the standardized detail.

    This exercises the handler directly (minimal Request scope) and ensures
    the JSON shape remains stable regardless of the original exception text.
    """
    from starlette.requests import Request
    from starlette.responses import JSONResponse

    scope = {
        "type": "http",
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": "/presign",
        "raw_path": b"/presign",
        "query_string": b"",
        "headers": [],
        "client": ("testclient", 50000),
        "server": ("testserver", 80),
    }

    request = Request(scope)

    # Original exception message that should be logged but not echoed back
    message = "Too many OCR requests"
    exc = type("_DummyRL", (), {"__str__": lambda _: message})()

    handler = limiter_mod._rate_limit_exceeded_handler_with_logging

    from unittest.mock import MagicMock

    mock_logger = MagicMock()
    monkeypatch.setattr(limiter_mod, "logger", mock_logger)

    response = handler(request, exc)  # type: ignore

    assert isinstance(response, JSONResponse)
    import json

    assert response.status_code == 429
    assert json.loads(response.body) == {"detail": "Rate limit exceeded. Please try again later."}

    # Confirm the original message was logged in the structured extra
    mock_logger.warning.assert_called()
    _, kwargs = mock_logger.warning.call_args
    extra = kwargs.get("extra", {})
    assert extra.get("detail") == message
