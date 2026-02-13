"""
Orchestration layer for document intelligence workflows.
Coordinates OCR pipelines with automated S3 storage integration.
"""

import asyncio
import hashlib
import json
import logging
import time
from typing import Any, Dict, Optional, cast

import redis.asyncio as redis
from fastapi import UploadFile

from ocr_service.exceptions import OCRPipelineError
from ocr_service.models import JobStatus
from ocr_service.services.storage import StorageService

from .ocr_engine import IterativeOCREngine

__all__ = ["OCRProcessor"]

logger = logging.getLogger("ocr-service.processor")


class OCRProcessor:
    """
    Orchestrates the OCR lifecycle, managing extraction and storage.
    """

    def __init__(
        self,
        ocr_engine: IterativeOCREngine,
        storage_service: StorageService,
        redis_client: redis.Redis,
    ):
        self.ocr_engine = ocr_engine
        self.storage_service = storage_service
        self.redis_client = redis_client

    async def close(self) -> None:
        """Cleanup processor resources."""
        await self.ocr_engine.close()

    async def process_file(
        self,
        file: UploadFile,
        reconstruct: bool = False,
        advanced: bool = False,
        doc_type: str = "generic",
        enable_reconstruction_config: bool = False,
        request_id: str = "N/A",
        redis_client: Optional[redis.Redis] = None,
        idempotency_key: Optional[str] = None,
        idempotency_ttl_seconds: int = 3600,
    ) -> Dict[str, Any]:
        """Handles UploadFile objects from FastAPI."""
        self._validate_file_type(file.content_type or "", file.filename, request_id)
        contents = await file.read()
        
        inferred_type = file.content_type
        if not inferred_type and file.filename:
            import mimetypes
            inferred_type = (
                mimetypes.guess_type(file.filename)[0] or "application/octet-stream"
            )

        return await self.process_bytes(
            contents=contents,
            filename=file.filename or "unknown",
            content_type=inferred_type or "application/octet-stream",
            reconstruct=reconstruct,
            advanced=advanced,
            doc_type=doc_type,
            enable_reconstruction_config=enable_reconstruction_config,
            request_id=request_id,
            redis_client=redis_client,
            idempotency_key=idempotency_key,
            idempotency_ttl_seconds=idempotency_ttl_seconds,
        )

    async def process_bytes(
        self,
        contents: bytes,
        filename: str,
        content_type: str,
        reconstruct: bool = False,
        advanced: bool = False,
        doc_type: str = "generic",
        enable_reconstruction_config: bool = False,
        request_id: str = "N/A",
        redis_client: Optional[redis.Redis] = None,
        idempotency_key: Optional[str] = None,
        idempotency_ttl_seconds: int = 3600,
    ) -> Dict[str, Any]:
        """Executes the full OCR pipeline on raw bytes."""
        start_time = time.time()
        redis_client = redis_client or self.redis_client
        
        cache_key = self._build_idempotency_key(
            idempotency_key=idempotency_key,
            filename=filename,
            doc_type=doc_type,
            reconstruct=reconstruct or enable_reconstruction_config,
            advanced=advanced,
            contents=contents,
        )

        # 1. Check Cache
        cached = await self._read_cached_result(redis_client, cache_key)
        if cached:
            if cached.get("status") == JobStatus.COMPLETED:
                return cast(Dict[str, Any], cached)
            if cached.get("status") == JobStatus.PROCESSING:
                raise OCRPipelineError(
                    phase="idempotency",
                    message="Request is already being processed",
                    status_code=409,
                    correlation_id=request_id,
                )

        # 2. Mark as processing
        await self._write_cached_result(
            redis_client, cache_key, {"status": JobStatus.PROCESSING, "request_id": request_id}, 120
        )

        try:
            # 3. Execute OCR
            use_recon = reconstruct or enable_reconstruction_config
            result = await self._execute_ocr_strategy(contents, advanced, use_recon, doc_type)

            if "error" in result:
                raise OCRPipelineError(
                    phase="extraction",
                    message=f"Extraction failure: {result['error']}",
                    status_code=400,
                    correlation_id=request_id,
                    filename=filename,
                )

            # 4. Persist to Storage
            s3_key = await self._persist_results(contents, filename, content_type, result, request_id)

            # 5. Finalize Result
            result.update({
                "status": JobStatus.COMPLETED,
                "filename": filename,
                "processing_time": round(time.time() - start_time, 3),
                "s3_key": s3_key,
                "request_id": request_id,
            })

            # 6. Cache final result
            await self._write_cached_result(redis_client, cache_key, result, idempotency_ttl_seconds)
            return result

        except Exception as e:
            await redis_client.delete(cache_key)
            if isinstance(e, OCRPipelineError):
                raise
            logger.exception("Internal processing failure")
            raise OCRPipelineError(
                phase="orchestration",
                message="Internal server error",
                status_code=500,
                correlation_id=request_id,
                filename=filename,
            ) from e

    def _validate_file_type(self, content_type: str, filename: Optional[str], request_id: str):
        if not content_type.startswith("image/"):
            raise OCRPipelineError(
                phase="validation",
                message="File must be a valid image format",
                status_code=400,
                correlation_id=request_id,
                filename=filename,
            )

    def _build_idempotency_key(
        self,
        idempotency_key: Optional[str],
        filename: str,
        doc_type: str,
        reconstruct: bool,
        advanced: bool,
        contents: bytes,
    ) -> str:
        if idempotency_key:
            return f"ocr:idempotency:{idempotency_key}"

        h = hashlib.sha256()
        h.update(contents)
        meta = f"{filename}|{doc_type}|{int(reconstruct)}|{int(advanced)}"
        h.update(meta.encode("utf-8"))
        return f"ocr:idempotency:{h.hexdigest()}"

    async def _read_cached_result(
        self, redis_client: redis.Redis, cache_key: str
    ) -> Optional[Dict[str, Any]]:
        try:
            raw = await redis_client.get(cache_key)
            return json.loads(raw) if raw else None
        except Exception:
            logger.exception("Redis read failed")
            return None

    async def _write_cached_result(
        self, redis_client: redis.Redis, cache_key: str, value: Dict[str, Any], ttl: int
    ) -> None:
        try:
            await redis_client.set(cache_key, json.dumps(value), ex=ttl)
        except Exception:
            logger.exception("Redis write failed")

    async def _execute_ocr_strategy(
        self, contents: bytes, advanced: bool, use_recon: bool, doc_type: str
    ) -> Dict[str, Any]:
        if advanced:
            return await self.ocr_engine.process_image_advanced(contents, doc_type=doc_type)
        return await self.ocr_engine.process_image(contents, use_reconstruction=use_recon)

    async def _persist_results(
        self,
        contents: bytes,
        filename: str,
        content_type: str,
        result: Dict[str, Any],
        request_id: str,
    ) -> str:
        try:
            # Upload raw file
            s3_key = await asyncio.to_thread(
                self.storage_service.upload_file, contents, filename, content_type
            )
            
            # Optional: upload metadata
            if result.get("reconstruction"):
                await asyncio.to_thread(
                    self.storage_service.upload_json, result["reconstruction"], filename
                )
            
            return s3_key
        except Exception as e:
            logger.error("Storage upload failed: %s", e)
            raise OCRPipelineError(
                phase="storage",
                message="Failed to persist results to storage",
                status_code=500,
                correlation_id=request_id,
                filename=filename,
            ) from e
