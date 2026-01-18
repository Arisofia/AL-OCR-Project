import logging
import time
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi.errors import RateLimitExceeded

from ocr_service.config import Settings, get_settings
from ocr_service.routers import ocr, storage, system
from ocr_service.utils.context import get_request_id_from_scope
from ocr_service.utils.custom_logging import setup_logging
from ocr_service.utils.limiter import _rate_limit_exceeded_handler_with_logging, limiter
from ocr_service.utils.monitoring import init_monitoring

logger = logging.getLogger("ocr-service")


def create_app(settings: Optional[Settings] = None) -> FastAPI:
    """
    Application factory for creating and configuring the FastAPI instance.
    """
    if settings is None:
        settings = get_settings()

    # Initialize enterprise-grade logging & monitoring
    setup_logging(level=logging.INFO)
    init_monitoring(settings, release=getattr(settings, "version", None))

    app = FastAPI(
        title=settings.app_name,
        description=settings.app_description,
        version=settings.version,
        docs_url="/docs" if settings.environment != "production" else None,
        redoc_url="/redoc" if settings.environment != "production" else None,
    )

    # Rate Limiting
    app.state.limiter = limiter
    app.add_exception_handler(
        RateLimitExceeded,
        _rate_limit_exceeded_handler_with_logging,  # type: ignore
    )

    # Middleware
    @app.middleware("http")
    async def add_process_time_and_logging(request: Request, call_next):
        """Logs request lifecycle and adds performance metadata to responses."""
        start_time = time.time()
        request_id = get_request_id_from_scope(request.scope)

        logger.info(
            "Request started | Path: %s | Method: %s | ID: %s",
            request.url.path,
            request.method,
            request_id,
        )

        response = await call_next(request)

        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = f"{process_time:.4f}s"
        response.headers["X-Request-ID"] = request_id

        logger.info(
            "Request finished | Path: %s | Status: %d | Latency: %.4fs | ID: %s",
            request.url.path,
            response.status_code,
            process_time,
            request_id,
        )

        return response

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Routers
    app.include_router(system.router, tags=["System"])
    app.include_router(ocr.router, tags=["OCR"])
    app.include_router(storage.router, tags=["Storage"])

    return app
