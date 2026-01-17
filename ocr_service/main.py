"""
Core API Gateway for the AL Financial OCR Service.
Simplified version using decoupled routers.
"""

import logging
import time

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from mangum import Mangum
from slowapi.errors import RateLimitExceeded

from ocr_service.config import get_settings
from ocr_service.routers import ocr, storage, system
from ocr_service.utils.custom_logging import setup_logging
from ocr_service.utils.limiter import _rate_limit_exceeded_handler_with_logging, limiter
from ocr_service.utils.monitoring import init_monitoring

# Initialize enterprise-grade logging
setup_logging(level=logging.INFO)
logger = logging.getLogger("ocr-service")

settings = get_settings()
init_monitoring(settings, release=getattr(settings, "version", None))

app = FastAPI(
    title=settings.app_name,
    description=settings.app_description,
    version=settings.version,
    docs_url="/docs" if settings.environment != "production" else None,
    redoc_url="/redoc" if settings.environment != "production" else None,
)
app.state.limiter = limiter
app.add_exception_handler(
    RateLimitExceeded,
    _rate_limit_exceeded_handler_with_logging,  # type: ignore
)


@app.middleware("http")
async def add_process_time_and_logging(request: Request, call_next):
    """Logs request lifecycle and adds performance metadata to responses."""
    start_time = time.time()

    # Try to extract request ID from ocr router helper if needed,
    # or just use a local one for middleware
    request_id = "local-development"
    scope = request.scope
    if "aws.context" in scope:
        request_id = str(
            getattr(scope["aws.context"], "aws_request_id", "local-development")
        )

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


# Global CORS Policy
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(system.router, tags=["System"])
app.include_router(ocr.router, tags=["OCR"])
app.include_router(storage.router, tags=["Storage"])

# AWS Lambda Integration
handler = Mangum(app)

if __name__ == "__main__":
    import uvicorn  # type: ignore

    uvicorn.run("ocr_service.main:app", host="0.0.0.0", port=8000, reload=True)
