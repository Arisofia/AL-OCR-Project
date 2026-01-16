from slowapi.errors import RateLimitExceeded
from ocr_service.utils import limiter as limiter_mod
from ocr_service.main import app


def test_rate_limit_handler_registered():
    """The FastAPI app should register our enhanced rate-limit handler."""
    handler = app.exception_handlers.get(RateLimitExceeded)
    assert handler is limiter_mod._rate_limit_exceeded_handler_with_logging
