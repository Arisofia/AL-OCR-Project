"""
Rate limiting utilities for the OCR Service.
Provides fallback mechanisms when slowapi is not available.
"""

import logging
from typing import Optional
from fastapi import Request
from fastapi.responses import JSONResponse
from opentelemetry import trace

# SlowAPI imports placed near other third-party imports
from slowapi import Limiter  # type: ignore
from slowapi.errors import RateLimitExceeded  # type: ignore
from slowapi.util import get_remote_address  # type: ignore

tracer = trace.get_tracer(__name__)

logger = logging.getLogger("ocr-service.limiter")

__all__ = [
    "Limiter",
    "RateLimitExceeded",
    "get_remote_address",
    "init_limiter",
    "_rate_limit_exceeded_handler_with_logging",
]


def _rate_limit_exceeded_handler_with_logging(
    request: Request, exc: RateLimitExceeded
) -> JSONResponse:
    """Enhanced rate limit handler with structured logging and tracing."""
    with tracer.start_as_current_span("limiter.rate_limit_exceeded"):
        logger.warning(
            "Rate limit exceeded",
            extra={
                "path": request.url.path,
                "remote_addr": get_remote_address(request),
                "detail": str(exc),
            },
        )
        return JSONResponse(
            status_code=429,
            content={"detail": str(exc)},
        )


def init_limiter() -> Optional[Limiter]:
    """Initializes and returns a Limiter instance with tracing.

    Returns:
        Optional[Limiter]: The initialized Limiter instance, or None if initialization
        fails.
    """
    with tracer.start_as_current_span("limiter.init_limiter"):
        try:
            limiter = Limiter(key_func=get_remote_address)
            logger.info("Limiter initialized successfully.")
            return limiter
        except Exception as e:
            logger.error(f"Failed to initialize Limiter: {e}")
            return None
