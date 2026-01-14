"""
Orchestration layer for document intelligence workflows and result persistence.
Coordinates synchronous and advanced OCR pipelines with automated S3 storage integration.
"""

import logging
import time
from typing import Dict, Any

from fastapi import UploadFile, HTTPException

from services.storage import StorageService
from .ocr_engine import IterativeOCREngine

logger = logging.getLogger("ocr-service.processor")


class OCRProcessor:
    """
    Main orchestrator for the OCR lifecycle, managing data extraction and storage synchronization.
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
    ) -> Dict[str, Any]:
        """
        Executes the full OCR pipeline: Validation, Extraction, and Cloud Persistence.
        """
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Protocol Violation: File must be a valid image format")

        start_time = time.time()
        contents = await file.read()

        try:
            # Execute targeted OCR strategy
            if advanced:
                result = await self.ocr_engine.process_image_advanced(
                    contents, doc_type=doc_type
                )
            else:
                use_recon = reconstruct or enable_reconstruction_config
                result = self.ocr_engine.process_image(
                    contents, use_reconstruction=use_recon
                )

            if "error" in result:
                raise HTTPException(status_code=400, detail=f"Extraction failure: {result['error']}")

            # Synchronize raw document and extracted intelligence to S3
            s3_key = self.storage_service.upload_file(
                content=contents, filename=file.filename, content_type=file.content_type
            )

            if result.get("reconstruction") and result["reconstruction"].get("meta"):
                self.storage_service.upload_json(
                    data=result["reconstruction"], filename=file.filename
                )

            processing_time = round(time.time() - start_time, 3)
            
            # Enrich response with traceability metadata
            result.update({
                "filename": file.filename,
                "processing_time": processing_time,
                "s3_key": s3_key
            })
            return result

        except HTTPException:
            raise
        except Exception as e:
            logger.error("Pipeline failure for %s | Error: %s", file.filename, e)
            raise HTTPException(
                status_code=500,
                detail="Internal processing failure in OCR orchestrator"
            ) from e
