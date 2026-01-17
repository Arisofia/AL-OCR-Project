from typing import Optional

from fastapi import Depends, HTTPException, Request, Security
from fastapi.security.api_key import APIKeyHeader

from ocr_service.config import Settings, get_settings
from ocr_service.modules.ocr_config import EngineConfig
from ocr_service.modules.ocr_engine import IterativeOCREngine
from ocr_service.modules.processor import OCRProcessor
from ocr_service.services.storage import StorageService

# Security and Identity Management
_settings_init = get_settings()
api_key_header = APIKeyHeader(name=_settings_init.api_key_header_name, auto_error=False)


async def get_api_key(
    header_value: Optional[str] = Security(api_key_header),
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
        return str(getattr(scope["aws.context"], "aws_request_id", "local-development"))
    return "local-development"


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
