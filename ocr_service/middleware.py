import logging
import time
import uuid
from collections.abc import Awaitable
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from ocr_service.utils.context import get_request_id_from_scope
from ocr_service.utils.tracing import get_current_trace_id

logger = logging.getLogger("ocr-service.middleware")


class ProcessTimeAndLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        start_time = time.time()

        request_id = get_request_id_from_scope(request.scope)
        correlation_id = (
            request.headers.get("X-Correlation-ID")
            or request.headers.get("X-Request-ID")
            or str(uuid.uuid4())
        )
        trace_id = get_current_trace_id()

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
        except Exception:
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
            response.headers["X-Trace-ID"] = str(trace_id)

        logger.info(
            "Request finished | Path: %s | Status: %d | Latency: %.4fs",
            request.url.path,
            response.status_code,
            process_time,
        )
        return response
