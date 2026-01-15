"""
Orchestration layer for document intelligence workflows.
Coordinates OCR pipelines with automated S3 storage integration.
"""

import asyncio
import logging
import time
from typing import Any, Dict

from fastapi import HTTPException, UploadFile
from services.storage import StorageService

from .ocr_engine import IterativeOCREngine

logger = logging.getLogger("ocr-service.processor")


class OCRProcessor:
    """
    Orchestrates the OCR lifecycle, managing extraction and storage.
    """

    def __init__(
        self,
        ocr_engine: IterativeOCREngine,
        storage_service: StorageService
    ):
        self.ocr_engine = ocr_engine
        self.storage_service = storage_service

    async def process(
        self,
        file: UploadFile,
        reconstruct: bool = False,
        advanced: bool = False,
        doc_type: str = "generic",
        enable_reconstruction_config: bool = False,
        request_id: str = "N/A",
    ) -> Dict[str, Any]:
        """
        Executes the full OCR pipeline: Validation, Extraction, and Cloud Persistence.
        """
        self._validate_file_type(file.content_type)

        start_time = time.time()
        contents = await file.read()

        try:
            # Execute targeted OCR strategy
            use_recon = reconstruct or enable_reconstruction_config
            result = await self._execute_ocr_strategy(
                contents, advanced, use_recon, doc_type
            )

            if "error" in result:
                raise HTTPException(
                    status_code=400,
                    detail=f"Extraction failure: {result['error']}"
                )

            # Synchronize raw document and extracted intelligence to S3
            s3_key = await self._persist_results(
                contents, file.filename, file.content_type, result
            )

            # Enrich response with traceability metadata
            result.update({
                "filename": file.filename,
                "processing_time": round(time.time() - start_time, 3),
                "s3_key": s3_key,
                "request_id": request_id
            })
            return result

        except HTTPException:
            raise
        except Exception as e:
            self._handle_pipeline_failure(file.filename, request_id, e)
            raise HTTPException(
                status_code=500,
                detail="Internal processing failure in OCR orchestrator"
            ) from e

    def _validate_file_type(self, content_type: str):
        """Ensures the uploaded file is an image."""
        if not content_type or not content_type.startswith("image/"):
            raise HTTPException(
                status_code=400,
                detail="File must be a valid image format"
            )

    async def _execute_ocr_strategy(
        self, contents: bytes, advanced: bool, use_recon: bool, doc_type: str
    ) -> Dict[str, Any]:
        """Selects and executes the appropriate OCR pipeline."""
        if advanced:
            return await self.ocr_engine.process_image_advanced(
                contents, doc_type=doc_type
            )

        return await asyncio.to_thread(
            self.ocr_engine.process_image,
            contents,
            use_reconstruction=use_recon,
        )

    async def _persist_results(
        self, contents: bytes, filename: str, content_type: str, result: Dict[str, Any]
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

        upload_results = await asyncio.gather(*upload_tasks)
        return upload_results[0]

    def _handle_pipeline_failure(
        self, filename: str, request_id: str, error: Exception
    ):
        """Logs pipeline failures with context."""
        logger.error("Pipeline failure for %s | Error: %s", filename, error)
        logger.error(
            "Pipeline failure | File: %s | RID: %s | Error: %s",
            filename,
            request_id,
            error,
        )
