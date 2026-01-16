"""
Rate limiting utilities for the OCR Service.
Provides fallback mechanisms when slowapi is not available.
"""

import logging
from typing import Any, Callable

from fastapi import Request
from fastapi.responses import JSONResponse

logger = logging.getLogger("ocr-service.limiter")

__all__ = [
    "Limiter",
    "RateLimitExceeded",
    "get_remote_address",
    "init_limiter",
    "_rate_limit_exceeded_handler_with_logging",
]

try:
    from slowapi import Limiter, _rate_limit_exceeded_handler  # type: ignore
    from slowapi.errors import RateLimitExceeded  # type: ignore
    from slowapi.util import get_remote_address  # type: ignore
except ImportError:  # pragma: no cover
    logger.warning("slowapi not found, using no-op rate limiter")

    class _NoopLimiter:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def limit(self, *_args: Any, **_kwargs: Any) -> Callable:
            def _decorator(f: Callable) -> Callable:
                return f

            return _decorator

    Limiter: Any = _NoopLimiter  # type: ignore[no-redef]

    def _no_rate_limit_handler(_request: Request, _exc: Exception) -> JSONResponse:
        """No-op rate limit handler for environments without slowapi."""
        return JSONResponse(
            status_code=429,
            content={"detail": "Rate limit exceeded (fallback handler)"},
        )

    _rate_limit_exceeded_handler: Any = _no_rate_limit_handler  # type: ignore[no-redef]
    RateLimitExceeded: Any = Exception  # type: ignore[no-redef]

    def get_remote_address(request: Request) -> str:
        """Extracts remote address from request or defaults to local host."""
        return getattr(getattr(request, "client", None), "host", "127.0.0.1")


def _rate_limit_exceeded_handler_with_logging(request: Request, exc: Exception) -> Any:
    """Enhanced rate limit handler with structured logging."""
    logger.warning(
        "Rate limit exceeded",
        extra={
            "path": request.url.path,
            "remote_addr": get_remote_address(request),
            "detail": str(exc),
        },
    )
    return _rate_limit_exceeded_handler(request, exc)  # type: ignore[arg-type]


def init_limiter() -> Limiter:
    """Initializes and returns a Limiter instance."""
    return Limiter(key_func=get_remote_address)
