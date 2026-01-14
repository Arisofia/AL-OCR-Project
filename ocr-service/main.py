"""
Core API Gateway for the AL Financial OCR Service.
Provides endpoints for document intelligence, pixel reconstruction, and S3 lifecycle management.
"""

import logging
import time

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
from mangum import Mangum
import boto3

from config import get_settings, Settings
from schemas import (
    OCRResponse,
    HealthResponse,
    ReconStatusResponse,
    PresignRequest,
    PresignResponse,
)
from services.storage import StorageService
from modules.ocr_engine import IterativeOCREngine
from modules.ocr_config import EngineConfig
from modules.processor import OCRProcessor

# Initialize enterprise-grade logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ocr-service")

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    description=settings.app_description,
    version=settings.version,
)

# Global CORS Policy
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
    raise HTTPException(status_code=403, detail="Unauthorized: Invalid or missing API Key")


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
async def perform_ocr(
    file: UploadFile = File(...),
    reconstruct: bool = False,
    advanced: bool = False,
    doc_type: str = "generic",
    _api_key: str = Depends(get_api_key),
    curr_settings: Settings = Depends(get_settings),
    processor: OCRProcessor = Depends(get_ocr_processor),
) -> OCRResponse:
    """
    Primary OCR entry point. 
    Processes uploaded documents with optional AI-driven pixel reconstruction.
    """
    result = await processor.process(
        file=file,
        reconstruct=reconstruct,
        advanced=advanced,
        doc_type=doc_type,
        enable_reconstruction_config=curr_settings.enable_reconstruction,
    )

    return OCRResponse(**result)


@app.post("/presign", response_model=PresignResponse)
async def generate_presigned_post(
    req: PresignRequest,
    _api_key: str = Depends(get_api_key),
    curr_settings: Settings = Depends(get_settings),
) -> PresignResponse:
    """
    Generates a secure, time-limited S3 POST URL for client-side direct uploads.
    Optimizes server bandwidth by offloading document transfer to AWS S3.
    """
    bucket = curr_settings.s3_bucket_name
    if not bucket:
        raise HTTPException(status_code=500, detail="Infrastructure failure: S3 bucket not configured")

    s3 = boto3.client("s3", region_name=curr_settings.aws_region)

    try:
        post = s3.generate_presigned_post(
            Bucket=bucket,
            Key=req.key,
            Fields={"Content-Type": req.content_type},
            Conditions=[["starts-with", "$Content-Type", req.content_type]],
            ExpiresIn=req.expires_in,
        )
    except Exception as exc:
        logger.exception("Failed to generate presigned credentials")
        raise HTTPException(
            status_code=500, detail="External service failure: Could not generate presigned post"
        ) from exc

    return PresignResponse(url=post["url"], fields=post["fields"])


# AWS Lambda Integration
handler = Mangum(app)

if __name__ == "__main__":
    import uvicorn  # type: ignore

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
