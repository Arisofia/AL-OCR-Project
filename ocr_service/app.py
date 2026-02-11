import logging
import time
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded

from ocr_service.config import Settings, get_settings
from ocr_service.exceptions import OCRPipelineError
from ocr_service.routers import ocr, storage, system
from ocr_service.schemas import ErrorResponse
from ocr_service.utils.context import get_request_id_from_scope
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

    @app.exception_handler(OCRPipelineError)
    async def ocr_pipeline_error_handler(request: Request, exc: OCRPipelineError):
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(
                phase=exc.phase,
                message=exc.message,
                correlation_id=exc.correlation_id,
                trace_id=exc.trace_id,
                filename=exc.filename,
            ).model_dump(exclude_none=True),
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

    # Startup checks for critical dependencies
    @app.on_event("startup")
    async def check_dependencies():
        """Perform lightweight dependency checks on startup and mark app state."""
        try:
            from ocr_service.utils.redis_factory import (
                get_redis_client,
                verify_redis_connection,
            )

            redis_client = get_redis_client(settings)
            redis_status = await verify_redis_connection(redis_client)
            if not redis_status.get("ok"):
                logger.error("Redis not available at startup: %s", redis_status)
                app.state.degraded = True
            else:
                app.state.degraded = False
            logger.info("Dependency check complete: %s", redis_status)
        except Exception as e:  # pragma: no cover - defensive
            logger.exception("Startup dependency check failed: %s", e)
            app.state.degraded = True

    # Routers
    app.include_router(system.router, tags=["System"])
    app.include_router(ocr.router, tags=["OCR"])
    app.include_router(storage.router, tags=["Storage"])

    return app
