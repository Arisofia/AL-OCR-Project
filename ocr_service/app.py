import logging
import time
import uuid
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded

from ocr_service.config import Settings, get_settings
from ocr_service.routers import ocr, storage, system
from ocr_service.schemas import ErrorContext, ErrorResponse
from ocr_service.utils.context import get_request_id_from_scope
from ocr_service.utils.limiter import _rate_limit_exceeded_handler_with_logging, limiter
from ocr_service.utils.monitoring import init_monitoring
from ocr_service.utils.redis_factory import (
    RedisInitializationError,
    get_redis_client,
    verify_redis_connection,
)

logger = logging.getLogger("ocr-service")


def _build_error_response(
    *,
    phase: str,
    detail: str,
    request: Request,
    status_code: int,
) -> JSONResponse:
    correlation_id = request.headers.get("X-Correlation-ID") or str(uuid.uuid4())
    trace_id = request.headers.get("X-Trace-ID") or get_request_id_from_scope(request.scope)
    payload = ErrorResponse(
        error=ErrorContext(
            phase=phase,
            correlation_id=correlation_id,
            trace_id=trace_id,
            detail=detail,
            filename=request.headers.get("X-File-Name"),
            content_type=request.headers.get("content-type"),
        )
    )
    return JSONResponse(status_code=status_code, content=payload.model_dump())


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

    @app.on_event("startup")
    async def startup_event() -> None:
        app.state.redis_client = get_redis_client(settings)
        await verify_redis_connection(app.state.redis_client, settings)

    @app.on_event("shutdown")
    async def shutdown_event() -> None:
        redis_client = getattr(app.state, "redis_client", None)
        if redis_client is not None:
            await redis_client.aclose()

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        logger.warning(
            "HTTP exception | path=%s | status=%d | detail=%s",
            request.url.path,
            exc.status_code,
            exc.detail,
        )
        return _build_error_response(
            phase="api",
            detail=str(exc.detail),
            request=request,
            status_code=exc.status_code,
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        logger.warning("Request validation failure | path=%s | detail=%s", request.url.path, exc)
        return _build_error_response(
            phase="validation",
            detail="Request validation failed",
            request=request,
            status_code=422,
        )

    @app.exception_handler(RedisInitializationError)
    async def redis_init_exception_handler(
        request: Request,
        exc: RedisInitializationError,
    ) -> JSONResponse:
        logger.error("Redis initialization failure | path=%s | error=%s", request.url.path, exc)
        return _build_error_response(
            phase="startup",
            detail=str(exc),
            request=request,
            status_code=503,
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
        correlation_id = request.headers.get("X-Correlation-ID") or str(uuid.uuid4())
        trace_id = request.headers.get("X-Trace-ID") or request_id

        logger.info(
            "Request started | path=%s | method=%s | request_id=%s | correlation_id=%s | trace_id=%s",
            request.url.path,
            request.method,
            request_id,
            correlation_id,
            trace_id,
        )

        response = await call_next(request)

        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = f"{process_time:.4f}s"
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Correlation-ID"] = correlation_id
        response.headers["X-Trace-ID"] = trace_id

        logger.info(
            "Request finished | path=%s | status=%d | latency=%.4fs | request_id=%s | correlation_id=%s | trace_id=%s",
            request.url.path,
            response.status_code,
            process_time,
            request_id,
            correlation_id,
            trace_id,
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
