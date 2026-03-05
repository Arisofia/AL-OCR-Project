"""
Rate limiting utilities for the OCR Service.
Provides fallback mechanisms when slowapi is not available.
"""

import logging
from typing import Optional

from fastapi import Request
from fastapi.responses import JSONResponse

try:
    from opentelemetry import trace

    tracer = trace.get_tracer(__name__)
except ImportError:

    class _NoopSpan:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class _NoopTracer:
        def start_as_current_span(self, _name: str):
            """Return a no-op span context manager when tracing is unavailable."""
            return _NoopSpan()

    tracer = _NoopTracer()

from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

logger = logging.getLogger("ocr-service.limiter")

__all__ = [
    "Limiter",
    "RateLimitExceeded",
    "_rate_limit_exceeded_handler_with_logging",
    "get_remote_address",
    "init_limiter",
]


def _rate_limit_exceeded_handler_with_logging(
    request: Request, exc: RateLimitExceeded
) -> JSONResponse:
    """Enhanced rate limit handler with structured logging and tracing."""
    with tracer.start_as_current_span("limiter.rate_limit_exceeded"):
        if not getattr(request.state, "rate_limit_logged", False):
            limit = getattr(request.state, "rate_limit", "unknown")
            logger.warning(
                "Rate limit exceeded | Path: %s | IP: %s | Limit: %s",
                request.url.path,
                get_remote_address(request),
                limit,
                extra={
                    "path": request.url.path,
                    "ip": get_remote_address(request),
                    "limit": limit,
                    "detail": str(exc),
                },
            )
            request.state.rate_limit_logged = True
        return JSONResponse(
            status_code=429,
            content={
                "detail": "Rate limit exceeded. Please try again later.",
            },
        )


def init_limiter() -> Optional[Limiter]:
    """Initializes and returns a Limiter instance with tracing.

    Returns:
        Optional[Limiter]: The initialized Limiter instance, or None if initialization
        fails.
    """
    with tracer.start_as_current_span("limiter.init_limiter"):
        try:
            limiter_instance = Limiter(key_func=get_remote_address)
            logger.info("Limiter initialized successfully.")
            return limiter_instance
        except Exception as e:
            logger.error("Failed to initialize Limiter: %s", e)
            return None


limiter = Limiter(key_func=get_remote_address)
