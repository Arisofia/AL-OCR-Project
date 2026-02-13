"""Module for document/image processors."""

import asyncio
import hashlib
import json
import logging
import mimetypes
import time
from typing import Any, Optional, cast

import redis.asyncio as redis
import redis.exceptions as redis_exceptions
from fastapi import UploadFile

from ocr_service.exceptions import OCRPipelineError
from ocr_service.metrics import (
    OCR_IDEMPOTENCY_HIT_COUNT,
    OCR_IDEMPOTENCY_MISS_COUNT,
    OCR_IDEMPOTENCY_REDIS_ERROR_COUNT,
)
from ocr_service.models import JobStatus
from ocr_service.services.storage import StorageService
from ocr_service.utils.tracing import get_current_trace_id

from .ocr_engine import IterativeOCREngine

__all__ = ["OCRProcessor"]

logger = logging.getLogger("ocr-service.processor")


class _NoopRedis:
    """Fallback Redis-like client for local runs without idempotency storage."""

    async def get(self, _key: str) -> None:
        return None

    async def set(
        self,
        _key: str,
        _value: str,
        ex: Optional[int] = None,
        _ex: Optional[int] = None,
    ) -> bool:
        _ = ex, _ex
        return True

    async def delete(self, _key: str) -> int:
        return 0


class OCRProcessor:
    """
    Orchestrates the OCR lifecycle, managing extraction and storage.
    """

    def __init__(
        self,
        ocr_engine: IterativeOCREngine,
        storage_service: StorageService,
        redis_client: Optional[redis.Redis] = None,
    ):
        """Initialize the OCR processor with engine and storage dependencies."""
        self.ocr_engine = ocr_engine
        self.storage_service = storage_service
        self.redis_client = redis_client or cast(redis.Redis, _NoopRedis())

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
    ) -> dict[str, Any]:
        """Handles UploadFile objects from FastAPI."""
        inferred_type = file.content_type
        if not inferred_type and file.filename:
            inferred_type = (
                mimetypes.guess_type(file.filename)[0] or "application/octet-stream"
            )

        self._validate_file_type(inferred_type or "", file.filename, request_id)
        contents = await file.read()

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
    ) -> dict[str, Any]:
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

        try:
            # 1. Check cache
            cached = await self._read_cached_result(redis_client, cache_key)
            if cached:
                OCR_IDEMPOTENCY_HIT_COUNT.inc()
                if cached.get("status") == JobStatus.COMPLETED:
                    return cast(dict[str, Any], cached)
                if cached.get("status") == JobStatus.PROCESSING:
                    raise OCRPipelineError(
                        phase="idempotency",
                        message="Request is already being processed",
                        status_code=409,
                        correlation_id=request_id,
                        trace_id=get_current_trace_id(),
                        filename=filename,
                    )
            else:
                OCR_IDEMPOTENCY_MISS_COUNT.inc()

            # 2. Mark as processing
            await self._write_cached_result(
                redis_client,
                cache_key,
                {"status": JobStatus.PROCESSING, "request_id": request_id},
                120,
            )

            # 3. Execute OCR
            use_recon = reconstruct or enable_reconstruction_config
            result = await self._execute_ocr_strategy(
                contents,
                advanced,
                use_recon,
                doc_type,
            )

            if "error" in result:
                raise OCRPipelineError(
                    phase="extraction",
                    message=f"Extraction failure: {result['error']}",
                    status_code=400,
                    correlation_id=request_id,
                    trace_id=get_current_trace_id(),
                    filename=filename,
                )

            # 4. Persist to storage
            s3_key = await self._persist_results(
                contents,
                filename,
                content_type,
                result,
                request_id,
            )

            # 5. Finalize result
            result.update(
                {
                    "status": JobStatus.COMPLETED,
                    "filename": filename,
                    "processing_time": round(time.time() - start_time, 3),
                    "s3_key": s3_key,
                    "request_id": request_id,
                }
            )

            # 6. Cache final result
            await self._write_cached_result(
                redis_client,
                cache_key,
                result,
                idempotency_ttl_seconds,
            )
            return result

        except OCRPipelineError as exc:
            if exc.phase != "idempotency" or exc.status_code != 409:
                await self._delete_cached_result(redis_client, cache_key)
            raise
        except (ValueError, TypeError) as exc:
            await self._delete_cached_result(redis_client, cache_key)
            logger.exception("Internal processing failure")
            raise OCRPipelineError(
                phase="orchestration",
                message="Internal server error",
                status_code=500,
                correlation_id=request_id,
                trace_id=get_current_trace_id(),
                filename=filename,
            ) from exc

    def _validate_file_type(
        self,
        content_type: str,
        filename: Optional[str],
        request_id: str,
    ) -> None:
        """Validate that the file is a supported image type."""
        if not content_type.startswith("image/"):
            raise OCRPipelineError(
                phase="validation",
                message="File must be a valid image format",
                status_code=400,
                correlation_id=request_id,
                trace_id=get_current_trace_id(),
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
        """Generate a unique key for idempotency based on request parameters."""
        if idempotency_key:
            return f"ocr:idempotency:{idempotency_key}"

        digest = hashlib.sha256()
        digest.update(contents)
        metadata = f"{filename}|{doc_type}|{int(reconstruct)}|{int(advanced)}"
        digest.update(metadata.encode("utf-8"))
        return f"ocr:idempotency:{digest.hexdigest()}"

    async def _read_cached_result(
        self,
        redis_client: redis.Redis,
        cache_key: str,
    ) -> Optional[dict[str, Any]]:
        """Retrieve cached OCR result from Redis."""
        try:
            raw = await redis_client.get(cache_key)
            return json.loads(raw) if raw else None
        except redis_exceptions.RedisError as exc:
            OCR_IDEMPOTENCY_REDIS_ERROR_COUNT.labels(operation="get").inc()
            logger.warning(
                "Redis unavailable for idempotency read; "
                "continuing in degraded mode: %s",
                exc,
            )
            return None
        except (json.JSONDecodeError, ValueError) as exc:
            OCR_IDEMPOTENCY_REDIS_ERROR_COUNT.labels(operation="get").inc()
            logger.exception("Redis read failed")
            raise OCRPipelineError(
                phase="idempotency",
                message="Failed to read idempotency state",
                status_code=500,
                trace_id=get_current_trace_id(),
            ) from exc

    async def _write_cached_result(
        self,
        redis_client: redis.Redis,
        cache_key: str,
        value: dict[str, Any],
        ttl: int,
    ) -> None:
        """Store OCR result in Redis with TTL."""
        try:
            serialized = json.dumps(value)
            await self._redis_set_with_ttl(redis_client, cache_key, serialized, ttl)
        except redis_exceptions.RedisError as exc:
            OCR_IDEMPOTENCY_REDIS_ERROR_COUNT.labels(operation="set").inc()
            logger.warning(
                "Redis unavailable for idempotency write; "
                "continuing in degraded mode: %s",
                exc,
            )
        except (TypeError, ValueError) as exc:
            OCR_IDEMPOTENCY_REDIS_ERROR_COUNT.labels(operation="set").inc()
            logger.exception("Redis write failed")
            raise OCRPipelineError(
                phase="idempotency",
                message="Failed to persist idempotency state",
                status_code=500,
                trace_id=get_current_trace_id(),
            ) from exc

    async def _redis_set_with_ttl(
        self,
        redis_client: redis.Redis,
        key: str,
        value: str,
        ttl: int,
    ) -> None:
        """Set a key-value pair in Redis with expiration time."""
        try:
            await redis_client.set(key, value, ex=ttl)
        except TypeError:
            # Some test doubles expose _ex instead of ex.
            await redis_client.set(key, value, _ex=ttl)  # type: ignore[call-arg]

    async def _delete_cached_result(
        self,
        redis_client: redis.Redis,
        cache_key: str,
    ) -> None:
        """Remove cached result from Redis."""
        try:
            await redis_client.delete(cache_key)
        except Exception:
            OCR_IDEMPOTENCY_REDIS_ERROR_COUNT.labels(operation="delete").inc()
            logger.exception("Redis delete failed")

    async def _execute_ocr_strategy(
        self,
        contents: bytes,
        advanced: bool,
        use_recon: bool,
        doc_type: str,
    ) -> dict[str, Any]:
        """Execute the appropriate OCR processing strategy."""
        if advanced:
            return await self.ocr_engine.process_image_advanced(
                contents, doc_type=doc_type
            )
        return await self.ocr_engine.process_image(
            contents,
            use_reconstruction=use_recon,
        )

    async def _persist_results(
        self,
        contents: bytes,
        filename: str,
        content_type: str,
        result: dict[str, Any],
        request_id: str,
    ) -> Optional[str]:
        """Persist OCR results and original file to storage."""
        try:
            # Upload raw file
            s3_key = await asyncio.to_thread(
                self.storage_service.upload_file,
                contents,
                filename,
                content_type,
            )

            # Optional metadata upload
            if result.get("reconstruction"):
                await asyncio.to_thread(
                    self.storage_service.upload_json,
                    result["reconstruction"],
                    filename,
                )

            return s3_key
        except Exception as exc:
            logger.error("Storage upload failed: %s", exc)
            raise OCRPipelineError(
                phase="storage",
                message="Failed to persist results to storage",
                status_code=500,
                correlation_id=request_id,
                trace_id=get_current_trace_id(),
                filename=filename,
            ) from exc
