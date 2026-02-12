import logging
import time
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from slowapi.errors import RateLimitExceeded

from ocr_service.config import Settings, get_settings
from ocr_service.exceptions import OCRPipelineError
from ocr_service.routers import ocr, storage, system
from ocr_service.schemas import ErrorResponse
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
    request_id = getattr(
        request.state, "request_id", get_request_id_from_scope(request.scope)
    )
    correlation_id = getattr(
        request.state,
        "correlation_id",
        request.headers.get("X-Correlation-ID") or request_id,
    )
    trace_id = getattr(
        request.state,
        "trace_id",
        request.headers.get("X-Trace-ID") or request_id,
    )
    payload = ErrorResponse(
        phase=phase,
        message=detail,
        correlation_id=correlation_id,
        trace_id=trace_id,
        filename=request.headers.get("X-File-Name"),
    )
    return JSONResponse(
        status_code=status_code, content=payload.model_dump(exclude_none=True)
    )


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
        # perform a light ping to verify connectivity; detailed checks run later
        try:
            redis_diag = await verify_redis_connection(app.state.redis_client)
            logger.info("Redis startup ping: %s", redis_diag)
        except Exception:
            logger.exception("Redis startup ping failed")

    @app.on_event("shutdown")
    async def shutdown_event() -> None:
        redis_client = getattr(app.state, "redis_client", None)
        if redis_client is not None:
            await redis_client.aclose()

    @app.exception_handler(HTTPException)
    async def http_exception_handler(
        request: Request, exc: HTTPException
    ) -> JSONResponse:
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
        logger.warning(
            "Request validation failure | path=%s | detail=%s", request.url.path, exc
        )
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
        logger.error(
            "Redis initialization failure | path=%s | error=%s", request.url.path, exc
        )
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

    @app.exception_handler(OCRPipelineError)
    async def ocr_pipeline_error_handler(_request: Request, exc: OCRPipelineError):
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

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, _exc: Exception):
        """Catch-all handler that converts unhandled exceptions into a
        structured `ErrorResponse` and logs contextual identifiers.
        """
        from ocr_service.utils.tracing import get_current_trace_id

        trace_id = get_current_trace_id()
        request_id = getattr(request.state, "request_id", None)
        correlation_id = getattr(request.state, "correlation_id", None)

        logger.exception(
            "Unhandled exception handled by generic handler | RID=%s CID=%s",
            request_id,
            correlation_id,
        )

        payload = ErrorResponse(
            phase="orchestration",
            message="Internal server error",
            correlation_id=correlation_id or request_id,
            trace_id=trace_id,
        )
        return JSONResponse(
            status_code=500, content=payload.model_dump(exclude_none=True)
        )

    # Metrics endpoint
    @app.get("/metrics")
    async def metrics():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    # Middleware
    @app.middleware("http")
    async def add_process_time_and_logging(request: Request, call_next):
        """Logs request lifecycle, injects correlation IDs and traces, and adds
        performance metadata to responses.
        """
        import uuid

        from ocr_service.utils.tracing import get_current_trace_id

        start_time = time.time()
        # Prefer AWS request id when present, otherwise accept incoming headers
        # or generate a UUID to correlate logs across systems.
        request_id = get_request_id_from_scope(request.scope)
        correlation_id = (
            request.headers.get("X-Correlation-ID")
            or request.headers.get("X-Request-ID")
            or str(uuid.uuid4())
        )

        trace_id = get_current_trace_id()

        # Expose context on request.state for downstream handlers and logs
        request.state.request_id = request_id
        request.state.correlation_id = correlation_id
        request.state.trace_id = trace_id

        logger.info(
            "Request started | Path: %s | Method: %s | RID: %s | CID: %s | TID: %s",
            request.url.path,
            request.method,
            request_id,
            correlation_id,
            trace_id,
        )

        try:
            response = await call_next(request)
        except Exception:  # Ensure we log context for uncaught errors
            logger.exception(
                "Unhandled exception during request processing | RID=%s CID=%s TID=%s",
                request_id,
                correlation_id,
                trace_id,
            )
            raise

        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = f"{process_time:.4f}s"
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Correlation-ID"] = correlation_id
        if trace_id is not None:
            response.headers["X-Trace-ID"] = trace_id

        logger.info(
            "Request finished | Path: %s | Status: %d | Latency: %.4fs",
            request.url.path,
            response.status_code,
            process_time,
        )
        logger.info(
            "Request context | RID: %s | CID: %s | TID: %s",
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

    # Startup checks for critical dependencies
    @app.on_event("startup")
    async def check_dependencies():
        """Perform lightweight dependency checks on startup and mark app state.

        If Redis is unavailable at startup the application marks itself as
        degraded. This function also records minimal diagnostics to aid
        troubleshooting in production.
        """
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

        # Add health check endpoint for convenience (non-production friendly)
        # and allow ops to probe internal readiness.
        app.state.redis_diagnostics = (
            redis_status if "redis_status" in locals() else {"ok": False}
        )

    # Routers
    app.include_router(system.router, tags=["System"])
    app.include_router(ocr.router, tags=["OCR"])
    app.include_router(storage.router, tags=["Storage"])

    return app
