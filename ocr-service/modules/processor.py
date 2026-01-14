"""
Processor module coordinating OCR tasks and storage.

This module provides the OCRProcessor class which acts as an orchestrator
between the OCR engine and the storage services.
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
    Orchestrates the OCR processing pipeline and results persistence.
    """

    def __init__(
        self,
        ocr_engine: IterativeOCREngine,
        storage_service: StorageService
    ):
        """
        Initializes the processor with an engine and storage service.
        """
        self.ocr_engine = ocr_engine
        self.storage_service = storage_service

    async def process(
        self,
        file: UploadFile,
        reconstruct: bool = False,
        advanced: bool = False,
        doc_type: str = "generic",
        enable_reconstruction_config: bool = False
    ) -> Dict[str, Any]:
        """
        Reads, processes, and saves the results of an uploaded file.
        """
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")

        start_time = time.time()
        contents = await file.read()

        try:
            if advanced:
                result = await self.ocr_engine.process_image_advanced(
                    contents,
                    doc_type=doc_type
                )
            else:
                use_recon = reconstruct or enable_reconstruction_config
                result = self.ocr_engine.process_image(
                    contents,
                    use_reconstruction=use_recon
                )

            if "error" in result:
                raise HTTPException(status_code=400, detail=result["error"])

            s3_key = self.storage_service.upload_file(
                content=contents,
                filename=file.filename,
                content_type=file.content_type
            )

            if result.get("reconstruction") and result["reconstruction"].get("meta"):
                self.storage_service.upload_json(
                    data=result["reconstruction"],
                    filename=file.filename
                )

            processing_time = round(time.time() - start_time, 3)
            result.update({
                "filename": file.filename,
                "processing_time": processing_time,
                "s3_key": s3_key
            })
            return result

        except HTTPException:
            raise
        except Exception as e:
            logger.error("Failed to process %s: %s", file.filename, e)
            raise HTTPException(
                status_code=500,
                detail="Internal processing error"
            ) from e
