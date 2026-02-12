"""
Orchestration layer for document intelligence workflows.
Coordinates OCR pipelines with automated S3 storage integration.
"""

import asyncio
import json
import logging
import time
from hashlib import sha256
from typing import Any

from fastapi import HTTPException, UploadFile

import redis.asyncio as redis

from ocr_service.services.storage import StorageService

from .ocr_engine import IterativeOCREngine

__all__ = ["OCRProcessor"]

logger = logging.getLogger("ocr-service.processor")


class OCRProcessor:
    """
    Orchestrates the OCR lifecycle, managing extraction and storage.
    """

    def __init__(self, ocr_engine: IterativeOCREngine, storage_service: StorageService):
        self.ocr_engine = ocr_engine
        self.storage_service = storage_service

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
        redis_client: redis.Redis | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        """
        Handles UploadFile objects from FastAPI and routes to core process_bytes.
        """
        self._validate_file_type(file.content_type or "")
        contents = await file.read()
        # Infer content_type from UploadFile or file signature
        inferred_type = file.content_type
        if not inferred_type and file.filename:
            import mimetypes

            inferred_type = (
                mimetypes.guess_type(file.filename)[0] or "application/octet-stream"
            )

        return await self.process_bytes(
            contents=contents,
            filename=file.filename or "unknown",
            content_type=inferred_type,
            reconstruct=reconstruct,
            advanced=advanced,
            doc_type=doc_type,
            enable_reconstruction_config=enable_reconstruction_config,
            request_id=request_id,
            redis_client=redis_client,
            idempotency_key=idempotency_key,
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
        redis_client: redis.Redis | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        """
        Executes the full OCR pipeline on raw bytes: Extraction and Cloud Persistence.
        """
        start_time = time.time()

        try:
            cache_key = self._build_idempotency_key(
                idempotency_key=idempotency_key,
                filename=filename,
                doc_type=doc_type,
                reconstruct=reconstruct or enable_reconstruction_config,
                advanced=advanced,
                contents=contents,
            )
            cached_result = await self._read_cached_result(redis_client, cache_key)
            if cached_result is not None:
                logger.info("Idempotency cache hit | key=%s | filename=%s | request_id=%s", cache_key, filename, request_id)
                return cached_result

            # Execute targeted OCR strategy
            use_recon = reconstruct or enable_reconstruction_config
            result = await self._execute_ocr_strategy(
                contents, advanced, use_recon, doc_type
            )

            if "error" in result:
                raise HTTPException(
                    status_code=400, detail=f"Extraction failure: {result['error']}"
                )

            # Synchronize raw document and extracted intelligence to S3
            s3_key = await self._persist_results(
                contents,
                filename,
                content_type,
                result,
            )

            # Enrich response with traceability metadata
            result.update(
                {
                    "filename": filename,
                    "processing_time": round(time.time() - start_time, 3),
                    "s3_key": s3_key,
                    "request_id": request_id,
                }
            )
            await self._write_cached_result(redis_client, cache_key, result)
            return result

        except HTTPException:
            raise
        except Exception as e:
            self._handle_pipeline_failure(filename, request_id, e)
            raise HTTPException(
                status_code=500,
                detail="Internal processing failure in OCR orchestrator",
            ) from e

    def _validate_file_type(self, content_type: str):
        """Ensures the uploaded file is an image."""
        if not content_type or not content_type.startswith("image/"):
            raise HTTPException(
                status_code=400, detail="File must be a valid image format"
            )

    async def _execute_ocr_strategy(
        self, contents: bytes, advanced: bool, use_recon: bool, doc_type: str
    ) -> dict[str, Any]:
        """Selects and executes the appropriate OCR pipeline."""
        if advanced:
            return await self.ocr_engine.process_image_advanced(
                contents, doc_type=doc_type
            )

        return await self.ocr_engine.process_image(
            contents,
            use_reconstruction=use_recon,
        )

    async def _persist_results(
        self, contents: bytes, filename: str, content_type: str, result: dict[str, Any]
    ) -> str:
        """Uploads files and metadata to cloud storage."""
        upload_tasks = [
            asyncio.to_thread(
                self.storage_service.upload_file,
                contents,
                filename,
                content_type,
            )
        ]

        if result.get("reconstruction") and result["reconstruction"].get("meta"):
            upload_tasks.append(
                asyncio.to_thread(
                    self.storage_service.upload_json,
                    result["reconstruction"],
                    filename,
                )
            )

        upload_results = await asyncio.gather(*upload_tasks, return_exceptions=True)
        for res in upload_results:
            if isinstance(res, Exception):
                logger.error("Storage upload failed: %s", res)
                raise HTTPException(
                    status_code=500, detail="Failed to persist results to storage"
                )
        return str(upload_results[0])

    def _handle_pipeline_failure(
        self, filename: str, request_id: str, error: Exception
    ):
        """Logs pipeline failures with context."""
        logger.error(
            "Pipeline failure | File: %s | RID: %s | Error: %s",
            filename,
            request_id,
            error,
        )


    def _build_idempotency_key(
        self,
        *,
        idempotency_key: str | None,
        filename: str,
        doc_type: str,
        reconstruct: bool,
        advanced: bool,
        contents: bytes,
    ) -> str:
        if idempotency_key:
            return f"ocr:idempotency:{idempotency_key}"

        digest = sha256(contents).hexdigest()
        return (
            f"ocr:idempotency:auto:{filename}:{doc_type}:{int(reconstruct)}:"
            f"{int(advanced)}:{digest}"
        )

    async def _read_cached_result(
        self, redis_client: redis.Redis | None, cache_key: str
    ) -> dict[str, Any] | None:
        if redis_client is None:
            return None

        try:
            cached = await redis_client.get(cache_key)
            if not cached:
                return None
            if isinstance(cached, bytes):
                cached = cached.decode("utf-8")
            return json.loads(cached)
        except Exception as exc:
            logger.exception(
                "Failed to read idempotency cache | key=%s | error=%s",
                cache_key,
                exc,
            )
            return None

    async def _write_cached_result(
        self, redis_client: redis.Redis | None, cache_key: str, result: dict[str, Any]
    ) -> None:
        if redis_client is None:
            return

        try:
            await redis_client.set(cache_key, json.dumps(result), ex=3600)
        except Exception as exc:
            logger.exception(
                "Failed to write idempotency cache | key=%s | error=%s",
                cache_key,
                exc,
            )
