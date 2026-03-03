"""Module for document/image processors."""

import asyncio
import hashlib
import json
import logging
import mimetypes
import time
from dataclasses import dataclass
from typing import Any, ClassVar, Optional, cast

import redis.asyncio as redis  # pylint: disable=import-error
import redis.exceptions as redis_exceptions  # pylint: disable=import-error
from fastapi import UploadFile  # pylint: disable=import-error

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
from .pdf_converter import is_pdf, pdf_pages_to_images

__all__ = ["OCRProcessor", "ProcessingConfig"]

logger = logging.getLogger("ocr-service.processor")
OCTET_STREAM = "application/octet-stream"


@dataclass
class ProcessingConfig:
    """Configuration for OCR processing operations."""

    reconstruct: bool = False
    advanced: bool = False
    doc_type: str = "generic"
    enable_reconstruction_config: bool = False
    request_id: str = "N/A"
    idempotency_key: Optional[str] = None
    idempotency_ttl_seconds: int = 3600


class _NoopRedis:
    """Fallback Redis-like client for local runs without idempotency storage."""

    async def get(self, _key: str) -> None:
        """Get method for compatibility."""
        await asyncio.sleep(0)
        return None  # noqa: RET501,PLR1711

    async def set(
        self,
        _key: str,
        _value: str,
        ex: Optional[int] = None,
        _ex: Optional[int] = None,
    ) -> bool:
        """Set method for compatibility."""
        _ = ex, _ex
        await asyncio.sleep(0)
        return True

    async def delete(self, _key: str) -> int:
        """Delete method for compatibility."""
        await asyncio.sleep(0)
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
        config: Optional[ProcessingConfig] = None,
        redis_client: Optional[redis.Redis] = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Handles UploadFile objects from FastAPI."""
        if config is None:
            config = ProcessingConfig(**kwargs)

        inferred_type = file.content_type
        if not inferred_type and file.filename:
            inferred_type = mimetypes.guess_type(file.filename)[0] or OCTET_STREAM

        self._validate_file_type(inferred_type or "", file.filename, config.request_id)
        contents = await file.read()

        return await self.process_bytes(
            contents=contents,
            filename=file.filename or "unknown",
            content_type=inferred_type or OCTET_STREAM,
            config=config,
            redis_client=redis_client,
        )

    async def process_bytes(  # pylint: disable=too-many-arguments
        self,
        contents: bytes,
        filename: str,
        content_type: str,
        config: Optional[ProcessingConfig] = None,
        redis_client: Optional[redis.Redis] = None,
    ) -> dict[str, Any]:
        """Executes the full OCR pipeline on raw bytes."""
        if config is None:
            config = ProcessingConfig()

        start_time = time.time()
        redis_client = redis_client or self.redis_client

        cache_key = self._build_idempotency_key(
            idempotency_key=config.idempotency_key,
            filename=filename,
            doc_type=config.doc_type,
            reconstruct=config.reconstruct or config.enable_reconstruction_config,
            advanced=config.advanced,
            contents=contents,
        )

        try:
            cached_result = await self._resolve_or_lock_idempotency(
                redis_client=redis_client,
                cache_key=cache_key,
                request_id=config.request_id,
                filename=filename,
            )
            if cached_result is not None:
                return cached_result

            result = await self._run_extraction(contents=contents, config=config)

            if "error" in result:
                raise OCRPipelineError(
                    phase="extraction",
                    message=f"Extraction failure: {result['error']}",
                    status_code=400,
                    correlation_id=config.request_id,
                    trace_id=get_current_trace_id(),
                    filename=filename,
                )

            # 4. Persist to storage
            s3_key = await self._persist_results(
                contents,
                filename,
                content_type,
                result,
                config.request_id,
            )

            # 5. Finalize result
            result.update(
                {
                    "status": JobStatus.COMPLETED,
                    "filename": filename,
                    "processing_time": round(time.time() - start_time, 3),
                    "s3_key": s3_key,
                    "request_id": config.request_id,
                }
            )

            # 6. Cache final result
            await self._write_cached_result(
                redis_client,
                cache_key,
                result,
                config.idempotency_ttl_seconds,
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
                correlation_id=config.request_id,
                trace_id=get_current_trace_id(),
                filename=filename,
            ) from exc

    _ACCEPTED_EXTENSIONS: ClassVar[frozenset[str]] = frozenset({
        ".pdf", ".tiff", ".tif", ".heic", ".heif", ".avif", ".webp", ".bmp",
    })

    async def _resolve_or_lock_idempotency(
        self,
        redis_client: redis.Redis,
        cache_key: str,
        request_id: str,
        filename: str,
    ) -> Optional[dict[str, Any]]:
        """Return cached result when complete, or lock request as processing."""
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

        await self._write_cached_result(
            redis_client,
            cache_key,
            {"status": JobStatus.PROCESSING, "request_id": request_id},
            120,
        )
        return None

    async def _run_extraction(
        self,
        contents: bytes,
        config: ProcessingConfig,
    ) -> dict[str, Any]:
        """Run OCR using PDF or image strategy depending on input contents."""
        use_recon = config.reconstruct or config.enable_reconstruction_config
        if is_pdf(contents):
            return await self._process_pdf(
                contents,
                config.advanced,
                use_recon,
                config.doc_type,
            )
        return await self._execute_ocr_strategy(
            contents,
            config.advanced,
            use_recon,
            config.doc_type,
        )

    def _validate_file_type(
        self,
        content_type: str,
        filename: Optional[str],
        request_id: str,
    ) -> None:
        """Validate that the file is a supported image or document type."""
        if content_type.startswith("image/"):
            return
        if content_type == "application/pdf":
            return
        if content_type == OCTET_STREAM and filename:
            ext = f".{filename.rsplit('.', 1)[-1].lower()}" if "." in filename else ""
            if ext in self._ACCEPTED_EXTENSIONS:
                return
        raise OCRPipelineError(
            phase="validation",
            message=(
                "Unsupported file type. Accepted formats: image/*, "
                "application/pdf, or application/octet-stream with a "
                ".pdf/.tiff/.tif/.heic/.heif/.avif/.webp/.bmp extension."
            ),
            status_code=400,
            correlation_id=request_id,
            trace_id=get_current_trace_id(),
            filename=filename,
        )

    def _build_idempotency_key(  # pylint: disable=too-many-arguments
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
        except ValueError as exc:
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
        except redis_exceptions.RedisError:
            OCR_IDEMPOTENCY_REDIS_ERROR_COUNT.labels(operation="delete").inc()
            logger.exception("Redis delete failed")

    async def _process_pdf(
        self,
        contents: bytes,
        advanced: bool,
        use_recon: bool,
        doc_type: str,
    ) -> dict[str, Any]:
        """Convert a PDF to images and run OCR on each page, then aggregate."""
        import cv2  # pylint: disable=import-outside-toplevel

        pages = pdf_pages_to_images(contents)
        if not pages:
            return {"error": "PDF produced no renderable pages"}

        texts: list[str] = []
        best_result: Optional[dict[str, Any]] = None
        best_confidence: float = -1.0
        combined_iterations: list[dict[str, Any]] = []

        for page_idx, page_array in enumerate(pages):
            ok, buf = cv2.imencode(".png", page_array)
            if not ok:
                continue
            page_bytes = buf.tobytes()
            page_result = await self._execute_ocr_strategy(
                page_bytes, advanced, use_recon, doc_type
            )
            if "error" in page_result:
                continue
            texts.append(page_result.get("text", ""))
            page_conf = float(page_result.get("confidence", 0.0))
            if page_conf > best_confidence:
                best_confidence = page_conf
                best_result = page_result
            for it in page_result.get("iterations", []) or []:
                it_copy = dict(it)
                it_copy["page"] = page_idx + 1
                combined_iterations.append(it_copy)

        if best_result is None:
            return {"error": "All PDF pages failed OCR"}

        best_result["text"] = "\n\n--- PAGE BREAK ---\n\n".join(
            t for t in texts if t
        )
        best_result["confidence"] = best_confidence
        best_result["iterations"] = combined_iterations
        return best_result

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
        try:
            return await self.ocr_engine.process_image(
                contents,
                use_reconstruction=use_recon,
                doc_type=doc_type,
            )
        except TypeError as exc:
            # Backward compatibility for test doubles that do not accept doc_type.
            if "unexpected keyword argument 'doc_type'" not in str(exc):
                raise
            return await self.ocr_engine.process_image(
                contents,
                use_reconstruction=use_recon,
            )

    async def _persist_results(  # pylint: disable=too-many-arguments
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
            # Try to catch storage-specific errors if available,
            # else fallback to Exception
            logger.error("Storage upload failed: %s", exc)
            raise OCRPipelineError(
                phase="storage",
                message="Failed to persist results to storage",
                status_code=500,
                correlation_id=request_id,
                trace_id=get_current_trace_id(),
                filename=filename,
            ) from exc
