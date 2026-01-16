"""
Core API Gateway for the AL Financial OCR Service.
Provides endpoints for document intelligence and S3 lifecycle management.
"""

import json
import logging
import os
import time
import uuid
from typing import Any, cast, Type  # type: ignore

from fastapi import Depends, FastAPI, File, HTTPException, Request, Security, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
from botocore.exceptions import ClientError
from mangum import Mangum
from pydantic import BaseModel
from redis import Redis  # type: ignore

from sentry_sdk.integrations.fastapi import FastApiIntegration

from ocr_service.config import Settings, get_settings
from ocr_service.modules.ocr_config import EngineConfig
from ocr_service.modules.ocr_engine import IterativeOCREngine
from ocr_service.modules.processor import OCRProcessor
from ocr_service.modules.active_learning_orchestrator import ALOrchestrator
from ocr_service.schemas import (
    HealthResponse,
    OCRResponse,
    PresignRequest,
    PresignResponse,
    ReconStatusResponse,
)
from ocr_service.services.storage import StorageService
from ocr_service.utils.limiter import (
    RateLimitExceeded,
    _rate_limit_exceeded_handler_with_logging as _rate_limit_exceeded_handler,
    init_limiter,
)
from ocr_service.utils.monitoring import init_monitoring

# Load settings
settings = get_settings()

# Initialize enterprise-grade monitoring (Logging + Sentry)
init_monitoring(
    settings,
    integrations=[FastApiIntegration()],
    release=f"ocr-service@{settings.version}",
)
logger = logging.getLogger("ocr-service")

limiter = init_limiter()
app = FastAPI(
    title=settings.app_name,
    description=settings.app_description,
    version=settings.version,
    docs_url="/docs" if settings.environment != "production" else None,
    redoc_url="/redoc" if settings.environment != "production" else None,
)
app.state.limiter = limiter
app.add_exception_handler(
    cast(Type[Exception], RateLimitExceeded), _rate_limit_exceeded_handler
)


def get_request_id(request: Request) -> str:
    """Extracts AWS Request ID from Mangum scope or defaults to local trace."""
    scope = request.scope
    if "aws.context" in scope:
        return cast(str, scope["aws.context"].aws_request_id)
    return "local-development"


@app.middleware("http")
async def add_process_time_and_logging(request: Request, call_next):
    """Logs request lifecycle and adds performance metadata to responses."""
    start_time = time.time()

    # Extract request ID for traceability and attach to state
    request_id = get_request_id(request)
    request.state.request_id = request_id

    logger.info(
        "Request started",
        extra={
            "path": request.url.path,
            "method": request.method,
            "request_id": request_id,
        },
    )

    try:
        response = await call_next(request)
    except Exception as exc:
        process_time = time.time() - start_time
        logger.error(
            "Request failed",
            extra={
                "path": request.url.path,
                "method": request.method,
                "latency": f"{process_time:.4f}s",
                "request_id": request_id,
                "error": str(exc),
            },
            exc_info=True,
        )
        raise

    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = f"{process_time:.4f}s"
    response.headers["X-Request-ID"] = request_id

    logger.info(
        "Request finished",
        extra={
            "path": request.url.path,
            "status": response.status_code,
            "latency": f"{process_time:.4f}s",
            "request_id": request_id,
        },
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


# Dependency Injection Providers


def get_storage_service(
    curr_settings: Settings = Depends(get_settings),
) -> StorageService:
    """Provides an authenticated StorageService instance for S3 operations."""
    return StorageService(
        bucket_name=curr_settings.s3_bucket_name, settings=curr_settings
    )


def get_ocr_engine(
    curr_settings: Settings = Depends(get_settings),
) -> IterativeOCREngine:
    """Provides a configured IterativeOCREngine for document analysis."""
    config = EngineConfig(
        max_iterations=curr_settings.ocr_iterations,
        enable_reconstruction=curr_settings.enable_reconstruction,
    )
    return IterativeOCREngine(config=config)


def get_ocr_processor(
    engine: IterativeOCREngine = Depends(get_ocr_engine),
    storage: StorageService = Depends(get_storage_service),
) -> OCRProcessor:
    """Orchestrates the high-level OCR processing pipeline."""
    return OCRProcessor(engine, storage)


def get_al_orchestrator(
    engine: IterativeOCREngine = Depends(get_ocr_engine),
) -> ALOrchestrator:
    """Provides an Active Learning orchestrator."""
    return ALOrchestrator(engine.learning_engine)


# Security and Identity Management
api_key_header = APIKeyHeader(name=settings.api_key_header_name, auto_error=False)


async def get_api_key(
    header_value: str = Security(api_key_header),
    curr_settings: Settings = Depends(get_settings),
) -> str:
    """Enforces API Key authentication for protected resources."""
    if header_value == curr_settings.ocr_api_key:
        return header_value
    raise HTTPException(
        status_code=403, detail="Unauthorized: Invalid or missing API Key"
    )


# Runtime Package Detection
try:
    import ocr_reconstruct as _ocr_reconstruct_pkg  # type: ignore

    RECON_PKG_AVAILABLE = True
    RECON_PKG_VERSION = getattr(_ocr_reconstruct_pkg, "__version__", "unknown")
except Exception:
    RECON_PKG_AVAILABLE = False
    RECON_PKG_VERSION = None


@app.get("/health", response_model=HealthResponse)
async def health_check(
    settings: Settings = Depends(get_settings),
    storage: StorageService = Depends(get_storage_service),
    engine: IterativeOCREngine = Depends(get_ocr_engine),
) -> HealthResponse:
    """Service availability heartbeat with dependency checks."""
    services = {
        "s3": "healthy" if storage.check_connection() else "unconfigured/degraded",
        "supabase": (
            "healthy"
            if engine.learning_engine.check_connection()
            else "unconfigured/degraded"
        ),
        "openai": "configured" if settings.openai_api_key else "missing",
        "gemini": "configured" if settings.gemini_api_key else "missing",
    }

    # If a critical service is degraded, we might want to flag the whole service
    status = "degraded" if services["s3"] == "unconfigured/degraded" else "healthy"

    return HealthResponse(
        status=status,
        timestamp=time.time(),
        environment=settings.environment,
        services=services,
    )


@app.get("/recon/status", response_model=ReconStatusResponse)
async def recon_status(
    curr_settings: Settings = Depends(get_settings),
) -> ReconStatusResponse:
    """Retrieves pixel reconstruction capability and versioning metadata."""
    return ReconStatusResponse(
        reconstruction_enabled=curr_settings.enable_reconstruction,
        package_installed=RECON_PKG_AVAILABLE,
        package_version=RECON_PKG_VERSION,
    )


@app.post("/ocr", response_model=OCRResponse)
@limiter.limit("10/minute")
async def perform_ocr(
    request: Request,
    file: UploadFile = File(...),
    reconstruct: bool = False,
    advanced: bool = False,
    doc_type: str = "generic",
    _api_key: str = Depends(get_api_key),
    request_id: str = Depends(get_request_id),
    curr_settings: Settings = Depends(get_settings),
    processor: OCRProcessor = Depends(get_ocr_processor),
) -> OCRResponse:
    """
    Primary OCR entry point.
    Processes uploaded documents with optional AI-driven pixel reconstruction.
    """
    # Used by SlowAPI decorator for rate limiting; intentionally not accessed
    del request
    result = await processor.process(
        file=file,
        reconstruct=reconstruct,
        advanced=advanced,
        doc_type=doc_type,
        enable_reconstruction_config=curr_settings.enable_reconstruction,
        request_id=request_id,
    )

    return OCRResponse(**result)


@app.post("/presign", response_model=PresignResponse)
@limiter.limit("5/minute")
async def generate_presigned_post(
    request: Request,
    req: PresignRequest,
    _api_key: str = Depends(get_api_key),
    storage: StorageService = Depends(get_storage_service),
) -> PresignResponse:
    """
    Generates a secure, time-limited S3 POST URL for client-side direct uploads.
    Optimizes server bandwidth by offloading document transfer to AWS S3.
    """
    # Used by SlowAPI decorator for rate limiting; intentionally not accessed
    del request

    try:
        post = storage.generate_presigned_post(
            key=req.key,
            content_type=req.content_type or "application/octet-stream",
            expires_in=req.expires_in or 3600,
        )
    except ClientError as exc:
        request_id = exc.response.get("ResponseMetadata", {}).get("RequestId", "N/A")
        logger.error("S3 Presign failed | RequestId: %s | Error: %s", request_id, exc)
        raise HTTPException(
            status_code=500, detail=f"S3 rejected request [{request_id}]"
        ) from exc
    except RuntimeError as exc:
        logger.error("Configuration error during presign: %s", exc)
        raise HTTPException(status_code=500, detail="S3 bucket not configured") from exc
    except Exception as exc:
        logger.exception("Unexpected failure during presign generation")
        raise HTTPException(
            status_code=500, detail="Could not generate presigned post"
        ) from exc

    return PresignResponse(url=post["url"], fields=post["fields"])


# Job submission and status endpoints
r = Redis(host=os.getenv("REDIS_HOST", "localhost"), port=6379, db=0)


class OCRRequest(BaseModel):
    image_url: str
    document_type: str = "invoice"


@app.post("/api/v1/extract")
async def submit_job(request: OCRRequest):
    job_id = str(uuid.uuid4())
    job_payload = {
        "id": job_id,
        "url": request.image_url,
        "status": "QUEUED",
    }
    r.set(f"job:{job_id}", json.dumps(job_payload))
    r.rpush("ocr_tasks", job_id)
    return {
        "job_id": job_id,
        "status": "QUEUED",
        "check_url": f"/api/v1/jobs/{job_id}",
    }


@app.get("/api/v1/jobs/{job_id}")
async def get_status(job_id: str):
    if not (data := r.get(f"job:{job_id}")):
        raise HTTPException(status_code=404, detail="Job not found")
    return json.loads(cast(Any, data))


@app.post("/al/trigger")
async def trigger_al_cycle(
    n_samples: int = 20,
    _api_key: str = Depends(get_api_key),
    orchestrator: ALOrchestrator = Depends(get_al_orchestrator),
):
    """Manually triggers an Active Learning cycle."""
    result = await orchestrator.run_cycle(n_samples=n_samples)
    if result.get("status") == "validation_failed":
        raise HTTPException(
            status_code=422, detail="Data validation failed during AL cycle"
        )
    return result


# AWS Lambda Integration
handler = Mangum(app)

if __name__ == "__main__":
    import uvicorn  # type: ignore

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)  # nosec B104
