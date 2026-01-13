"""
Main entry point for the OCR FastAPI service.
"""

import logging
import time
from typing import Any

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Security, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
from mangum import Mangum

from config import get_settings, Settings
from schemas import OCRResponse, HealthResponse, ReconStatusResponse
from services.storage import StorageService
from modules.ocr_engine import IterativeOCREngine
from modules.ocr_config import EngineConfig
from modules.processor import OCRProcessor
import boto3
from typing import Dict
from schemas import PresignRequest, PresignResponse

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ocr-service")

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    description=settings.app_description,
    version=settings.version
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Dependency Providers
def get_storage_service(
    curr_settings: Settings = Depends(get_settings)
) -> StorageService:
    """
    Dependency provider for StorageService.
    """
    return StorageService(
        bucket_name=curr_settings.s3_bucket_name,
        settings=curr_settings
    )


def get_ocr_engine(
    curr_settings: Settings = Depends(get_settings)
) -> IterativeOCREngine:
    """
    Dependency provider for IterativeOCREngine.
    """
    config = EngineConfig(
        max_iterations=curr_settings.ocr_iterations,
        enable_reconstruction=curr_settings.enable_reconstruction
    )
    return IterativeOCREngine(config=config)


def get_ocr_processor(
    engine: IterativeOCREngine = Depends(get_ocr_engine),
    storage: StorageService = Depends(get_storage_service)
) -> OCRProcessor:
    """
    Dependency provider for OCRProcessor.
    """
    return OCRProcessor(engine, storage)


# Security
api_key_header = APIKeyHeader(name=settings.api_key_header_name, auto_error=False)


async def get_api_key(
    header_value: str = Security(api_key_header),
    curr_settings: Settings = Depends(get_settings)
) -> str:
    """
    Validates the API key from the request header.
    """
    if header_value == curr_settings.ocr_api_key:
        return header_value
    raise HTTPException(status_code=403, detail="Invalid API Key")


# Recon package availability (best-effort detection)
try:
    import ocr_reconstruct as _ocr_reconstruct_pkg  # type: ignore
    RECON_PKG_AVAILABLE = True
    RECON_PKG_VERSION = getattr(_ocr_reconstruct_pkg, "__version__", "unknown")
except (ImportError, Exception):
    RECON_PKG_AVAILABLE = False
    RECON_PKG_VERSION = None


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Health check endpoint to verify service status.
    """
    return HealthResponse(status="healthy", timestamp=time.time())


@app.get("/recon/status", response_model=ReconStatusResponse)
async def recon_status(
    curr_settings: Settings = Depends(get_settings)
) -> ReconStatusResponse:
    """
    Return reconstruction availability and package metadata.
    """
    return ReconStatusResponse(
        reconstruction_enabled=curr_settings.enable_reconstruction,
        package_installed=RECON_PKG_AVAILABLE,
        package_version=RECON_PKG_VERSION
    )


@app.post("/ocr", response_model=OCRResponse)
async def perform_ocr(
    file: UploadFile = File(...),
    reconstruct: bool = False,
    advanced: bool = False,
    doc_type: str = "generic",
    api_key: str = Depends(get_api_key),
    curr_settings: Settings = Depends(get_settings),
    processor: OCRProcessor = Depends(get_ocr_processor)
) -> OCRResponse:
    """
    Performs OCR on the uploaded file with optional reconstruction.
    """
    result = await processor.process(
        file=file,
        reconstruct=reconstruct,
        advanced=advanced,
        doc_type=doc_type,
        enable_reconstruction_config=curr_settings.enable_reconstruction
    )

    return OCRResponse(**result)


@app.post("/presign", response_model=PresignResponse)
async def generate_presigned_post(
    req: PresignRequest,
    api_key: str = Depends(get_api_key),
    curr_settings: Settings = Depends(get_settings),
) -> PresignResponse:
    """
    Generates a presigned S3 POST for direct uploads to the configured bucket.

    Client should POST form fields + file directly to `url` with the returned fields.
    """
    bucket = curr_settings.s3_bucket_name
    if not bucket:
        raise HTTPException(status_code=500, detail="S3 bucket not configured")

    s3 = boto3.client("s3", region_name=curr_settings.aws_region)

    try:
        post = s3.generate_presigned_post(
            Bucket=bucket,
            Key=req.key,
            Fields={"Content-Type": req.content_type},
            Conditions=[["starts-with", "$Content-Type", req.content_type]],
            ExpiresIn=req.expires_in,
        )
    except Exception as exc:  # pragma: no cover - Boto3 behavior
        logger.exception("Failed to generate presigned post")
        raise HTTPException(status_code=500, detail="Failed to generate presigned post")

    return PresignResponse(url=post["url"], fields=post["fields"])

# Lambda Handler
handler = Mangum(app)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
