"""
Core API Gateway for the AL Financial OCR Service.
Provides endpoints for document intelligence and S3 lifecycle management.
"""

import logging
import time

import boto3
from botocore.exceptions import ClientError
from config import Settings, get_settings
from fastapi import (Depends, FastAPI, File, HTTPException, Request, Security,
                     UploadFile)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
from mangum import Mangum
from modules.ocr_config import EngineConfig
from modules.ocr_engine import IterativeOCREngine
from modules.processor import OCRProcessor
from schemas import (HealthResponse, OCRResponse, PresignRequest,
                     PresignResponse, ReconStatusResponse)
from services.storage import StorageService
# slowapi is optional in test environments; silence static import errors for linters
# pylint: disable=import-error
try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.errors import RateLimitExceeded
    from slowapi.util import get_remote_address
except Exception:  # pragma: no cover - slowapi optional in tests
    class _NoopLimiter:
        def __init__(self, *_args, **_kwargs):
            pass

        def limit(self, *_args, **_kwargs):
            def _decorator(f):
                return f

            return _decorator

    Limiter = _NoopLimiter

    def _no_rate_limit_handler(_request, _exc):
        """No-op rate limit handler for environments without slowapi."""
        return None

    _rate_limit_exceeded_handler = _no_rate_limit_handler
    RateLimitExceeded = Exception

    def get_remote_address(request):
        return getattr(getattr(request, "client", None), "host", "127.0.0.1")

# Initialize enterprise-grade logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ocr-service")

settings = get_settings()

limiter = Limiter(key_func=get_remote_address)
app = FastAPI(
    title=settings.app_name,
    description=settings.app_description,
    version=settings.version,
    docs_url="/docs" if settings.environment != "production" else None,
    redoc_url="/redoc" if settings.environment != "production" else None,
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.middleware("http")
async def add_process_time_and_logging(request: Request, call_next):
    """Logs request lifecycle and adds performance metadata to responses."""
    start_time = time.time()

    # Extract request ID for traceability
    request_id = get_request_id(request)

    logger.info("Request started | Path: %s | Method: %s | ID: %s",
                request.url.path, request.method, request_id)

    response = await call_next(request)

    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = f"{process_time:.4f}s"
    response.headers["X-Request-ID"] = request_id

    logger.info("Request finished | Path: %s | Status: %d | Latency: %.4fs | ID: %s",
                request.url.path, response.status_code, process_time, request_id)

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


def get_request_id(request: Request) -> str:
    """Extracts AWS Request ID from Mangum scope or defaults to local trace."""
    scope = request.scope
    if "aws.context" in scope:
        return scope["aws.context"].aws_request_id
    return "local-development"


# Runtime Package Detection
try:
    import ocr_reconstruct as _ocr_reconstruct_pkg  # type: ignore

    RECON_PKG_AVAILABLE = True
    RECON_PKG_VERSION = getattr(_ocr_reconstruct_pkg, "__version__", "unknown")
except (ImportError, Exception):
    RECON_PKG_AVAILABLE = False
    RECON_PKG_VERSION = None


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Service availability heartbeat."""
    return HealthResponse(status="healthy", timestamp=time.time())


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
    curr_settings: Settings = Depends(get_settings),
) -> PresignResponse:
    """
    Generates a secure, time-limited S3 POST URL for client-side direct uploads.
    Optimizes server bandwidth by offloading document transfer to AWS S3.
    """
    # Used by SlowAPI decorator for rate limiting; intentionally not accessed
    del request
    bucket = curr_settings.s3_bucket_name
    if not bucket:
        raise HTTPException(
            status_code=500, detail="S3 bucket not configured"
        )

    s3 = boto3.client("s3", region_name=curr_settings.aws_region)

    try:
        post = s3.generate_presigned_post(
            Bucket=bucket,
            Key=req.key,
            Fields={"Content-Type": req.content_type},
            Conditions=[["starts-with", "$Content-Type", req.content_type]],
            ExpiresIn=req.expires_in,
        )
    except ClientError as exc:
        request_id = exc.response.get("ResponseMetadata", {}).get("RequestId", "N/A")
        logger.error("S3 Presign failed | RequestId: %s | Error: %s", request_id, exc)
        raise HTTPException(
            status_code=500,
            detail=f"S3 rejected request [{request_id}]"
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected failure during presign generation")
        raise HTTPException(
            status_code=500,
            detail="Could not generate presigned post"
        ) from exc

    return PresignResponse(url=post["url"], fields=post["fields"])


# AWS Lambda Integration
handler = Mangum(app)

if __name__ == "__main__":
    import uvicorn  # type: ignore

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
