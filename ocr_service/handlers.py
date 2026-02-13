import logging

from fastapi import HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded

from ocr_service.exceptions import OCRPipelineError
from ocr_service.schemas import ErrorResponse
from ocr_service.utils.context import get_request_id_from_scope
from ocr_service.utils.limiter import _rate_limit_exceeded_handler_with_logging
from ocr_service.utils.redis_factory import RedisInitializationError
from ocr_service.utils.tracing import get_current_trace_id

logger = logging.getLogger("ocr-service.handlers")


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
    content = payload.model_dump(exclude_none=False)
    content["detail"] = detail
    return JSONResponse(status_code=status_code, content=content)


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


async def ocr_pipeline_error_handler(
    request: Request, exc: OCRPipelineError
) -> JSONResponse:
    request_id = getattr(
        request.state, "request_id", get_request_id_from_scope(request.scope)
    )
    correlation_id = exc.correlation_id or getattr(
        request.state,
        "correlation_id",
        request.headers.get("X-Correlation-ID") or request_id,
    )
    trace_id = exc.trace_id or getattr(
        request.state,
        "trace_id",
        request.headers.get("X-Trace-ID"),
    )
    content = ErrorResponse(
        phase=exc.phase,
        message=exc.message,
        correlation_id=correlation_id,
        trace_id=trace_id,
        filename=exc.filename,
    ).model_dump(exclude_none=False)
    content["detail"] = exc.message
    return JSONResponse(
        status_code=exc.status_code,
        content=content,
    )


async def generic_exception_handler(request: Request, _exc: Exception) -> JSONResponse:
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
    content = payload.model_dump(exclude_none=False)
    content["detail"] = "Internal server error"
    return JSONResponse(status_code=500, content=content)


def register_handlers(app):
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(RedisInitializationError, redis_init_exception_handler)
    app.add_exception_handler(OCRPipelineError, ocr_pipeline_error_handler)
    app.add_exception_handler(Exception, generic_exception_handler)
    app.add_exception_handler(
        RateLimitExceeded, _rate_limit_exceeded_handler_with_logging
    )
