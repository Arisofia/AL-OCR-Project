"""
Orchestration layer for document intelligence workflows.
Coordinates OCR pipelines with automated S3 storage integration.
"""

import asyncio
import hashlib
import json
import logging
import time
from typing import Any, Optional, cast

import redis.asyncio as redis
from fastapi import UploadFile
from opentelemetry import trace

from ocr_service.exceptions import OCRPipelineError
from ocr_service.services.storage import StorageService

from .ocr_engine import IterativeOCREngine

__all__ = ["OCRProcessor"]

logger = logging.getLogger("ocr-service.processor")
tracer = trace.get_tracer(__name__)


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
    ) -> dict[str, Any]:
        """
        Handles UploadFile objects from FastAPI and routes to core process_bytes.
        """
        self._validate_file_type(file.content_type or "", file.filename, request_id)
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
    ) -> dict[str, Any]:
        """
        Executes the full OCR pipeline on raw bytes: Extraction and Cloud Persistence.
        """
        start_time = time.time()

        span = trace.get_current_span()
        trace_id: Optional[str] = None
        ctx = span.get_span_context() if span is not None else None
        if ctx is not None and getattr(ctx, "trace_id", None) is not None:
            trace_id = format(ctx.trace_id, "x")

        idempotency_key = (
            f"idempotency:{hashlib.sha256(contents).hexdigest()}:{filename}"
        )
        try:
            try:
                cached_result = await self.redis_client.get(idempotency_key)
            except Exception as e:  # pragma: no cover - redis defensive
                from ocr_service.metrics import OCR_IDEMPOTENCY_REDIS_ERROR_COUNT

                logger.exception("Redis GET failed during idempotency check: %s", e)
                OCR_IDEMPOTENCY_REDIS_ERROR_COUNT.labels(operation="get").inc()
                raise OCRPipelineError(
                    phase="idempotency",
                    message="Failed to query idempotency store",
                    status_code=500,
                    correlation_id=request_id,
                    trace_id=trace_id,
                    filename=filename,
                ) from e

            if cached_result:
                try:
                    # Redis may return bytes or str depending on client config.
                    # Normalize to text before JSON parsing.
                    if isinstance(cached_result, (bytes, bytearray)):
                        cached_text = cached_result.decode("utf-8")
                    else:
                        cached_text = cached_result

                    cached_data = cast(dict[str, Any], json.loads(cached_text))
                    if cached_data.get("status") == "processing":
                        raise OCRPipelineError(
                            phase="idempotency",
                            message="Duplicate request already being processed",
                            status_code=409,
                            correlation_id=request_id,
                            trace_id=trace_id,
                            filename=filename,
                        )
                    return cached_data
                except json.JSONDecodeError:
                    logger.warning(
                        "Could not decode cached idempotency result for key %s",
                        idempotency_key,
                    )

            try:
                await self.redis_client.set(
                    idempotency_key,
                    json.dumps({"status": "processing"}),
                    ex=120,  # 2 minutes TTL for processing
                )
            except Exception as e:  # pragma: no cover - redis defensive
                from ocr_service.metrics import OCR_IDEMPOTENCY_REDIS_ERROR_COUNT

                logger.exception("Redis SET failed during idempotency set: %s", e)
                OCR_IDEMPOTENCY_REDIS_ERROR_COUNT.labels(operation="set").inc()
                raise OCRPipelineError(
                    phase="idempotency",
                    message="Failed to set idempotency key",
                    status_code=500,
                    correlation_id=request_id,
                    trace_id=trace_id,
                    filename=filename,
                ) from e

            # Execute targeted OCR strategy
            use_recon = reconstruct or enable_reconstruction_config
            result = await self._execute_ocr_strategy(
                contents, advanced, use_recon, doc_type
            )

            if "error" in result:
                try:
                    await self.redis_client.delete(idempotency_key)
                except Exception as e:  # pragma: no cover - defensive
                    logger.exception(
                        "Redis DELETE failed when cleaning idempotency key: %s", e
                    )
                raise OCRPipelineError(
                    phase="extraction",
                    message=f"Extraction failure: {result['error']}",
                    status_code=400,
                    correlation_id=request_id,
                    trace_id=trace_id,
                    filename=filename,
                )

            # Synchronize raw document and extracted intelligence to S3
            s3_key = await self._persist_results(
                contents,
                filename,
                content_type,
                result,
                request_id,
                trace_id,
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

            await self.redis_client.set(
                idempotency_key,
                json.dumps(result),
                ex=3600,  # 1 hour TTL for final result
            )
            return result

        except OCRPipelineError as e:
            # Do not delete key on 409 conflict
            if e.status_code != 409:
                try:
                    await self.redis_client.delete(idempotency_key)
                except Exception as de:
                    from ocr_service.metrics import OCR_IDEMPOTENCY_REDIS_ERROR_COUNT

                    logger.exception(
                        "Redis DELETE failed during cleanup after OCRPipelineError: %s",
                        de,
                    )
                    OCR_IDEMPOTENCY_REDIS_ERROR_COUNT.labels(operation="delete").inc()
            raise
        except Exception as e:
            try:
                await self.redis_client.delete(idempotency_key)
            except Exception as de:
                logger.exception(
                    "Redis DELETE failed during cleanup after exception: %s", de
                )
            self._handle_pipeline_failure(filename, request_id, e)
            raise OCRPipelineError(
                phase="orchestration",
                message="Internal processing failure in OCR orchestrator",
                status_code=500,
                correlation_id=request_id,
                trace_id=trace_id,
                filename=filename,
            ) from e

    def _validate_file_type(
        self, content_type: str, filename: "Optional[str]", request_id: str
    ):
        """Ensures the uploaded file is an image."""
        if not content_type or not content_type.startswith("image/"):
            span = trace.get_current_span()
            trace_id: Optional[str] = None
            ctx = span.get_span_context() if span is not None else None
            if ctx is not None and getattr(ctx, "trace_id", None) is not None:
                trace_id = format(ctx.trace_id, "x")
            raise OCRPipelineError(
                phase="validation",
                message="File must be a valid image format",
                status_code=400,
                correlation_id=request_id,
                trace_id=trace_id,
                filename=filename,
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
        self,
        contents: bytes,
        filename: str,
        content_type: str,
        result: dict[str, Any],
        request_id: str,
        trace_id: Optional[str],
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
                raise OCRPipelineError(
                    phase="storage",
                    message="Failed to persist results to storage",
                    status_code=500,
                    correlation_id=request_id,
                    trace_id=trace_id,
                    filename=filename,
                )
        s3_key_result = upload_results[0]
        return s3_key_result if isinstance(s3_key_result, str) else "unknown"

    def _handle_pipeline_failure(
        self, filename: str, request_id: str, _error: Exception
    ):
        """Logs pipeline failures with context."""
        logger.exception(
            "Pipeline failure | File: %s | RID: %s",
            filename,
            request_id,
        )
