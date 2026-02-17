from typing import Optional

import redis.asyncio as redis
from fastapi import Depends, HTTPException, Request, Security
from fastapi.security.api_key import APIKeyHeader

from ocr_service.config import Settings, get_settings
from ocr_service.modules.ocr_config import EngineConfig
from ocr_service.modules.ocr_engine import IterativeOCREngine
from ocr_service.modules.processor import OCRProcessor
from ocr_service.services.storage import StorageService
from ocr_service.utils.context import get_request_id_from_scope
from ocr_service.utils.redis_factory import get_redis_client as create_redis_client

# Security and Identity Management
api_key_header = APIKeyHeader(name="X-API-KEY", auto_error=False)
dataset_key_header = APIKeyHeader(name="X-DATASET-KEY", auto_error=False)


async def get_api_key(
    header_value: Optional[str] = Security(api_key_header),
    curr_settings: Settings = Depends(get_settings),
) -> str:
    """Enforces API Key authentication for protected resources."""
    # Ensure api_key_header name matches settings if dynamic name is needed
    if header_value == curr_settings.ocr_api_key:
        return header_value
    raise HTTPException(
        status_code=403, detail="Unauthorized: Invalid or missing API Key"
    )


def get_request_id(request: Request) -> str:
    """Extracts AWS Request ID from Mangum scope or defaults to local trace."""
    return get_request_id_from_scope(request.scope)


async def get_dataset_upload_key(
    header_value: Optional[str] = Security(dataset_key_header),
    curr_settings: Settings = Depends(get_settings),
) -> str:
    """Enforces Dataset upload key for protected dataset endpoints."""
    expected = (curr_settings.dataset_upload_key or "").strip()
    if not expected:
        raise HTTPException(status_code=503, detail="Dataset uploads are disabled")
    if header_value == expected:
        return header_value
    raise HTTPException(
        status_code=403, detail="Unauthorized: Invalid or missing Dataset key"
    )


def get_redis_client_dep(
    settings: Settings = Depends(get_settings),
) -> redis.Redis:
    """Provides a Redis client instance."""
    return create_redis_client(settings)


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
        enable_bin_lookup=curr_settings.enable_bin_lookup,
    )
    return IterativeOCREngine(config=config)


def get_ocr_processor(
    engine: IterativeOCREngine = Depends(get_ocr_engine),
    storage: StorageService = Depends(get_storage_service),
    redis_client: redis.Redis = Depends(get_redis_client_dep),
) -> OCRProcessor:
    """Orchestrates the high-level OCR processing pipeline."""
    return OCRProcessor(engine, storage, redis_client)
