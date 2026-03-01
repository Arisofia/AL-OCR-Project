import logging
import time
from typing import Optional

from fastapi import APIRouter, Depends, File, Header, Request, UploadFile

from ocr_service.config import Settings, get_settings
from ocr_service.exceptions import OCRPipelineError
from ocr_service.metrics import OCR_ERROR_COUNT, OCR_REQUEST_COUNT, OCR_REQUEST_LATENCY
from ocr_service.modules.personal_doc_extractor import PersonalDocExtractor, detect_metadata
from ocr_service.modules.processor import OCRProcessor, ProcessingConfig
from ocr_service.routers.deps import get_api_key, get_ocr_processor, get_request_id
from ocr_service.schemas import DocumentField, DocumentResponse, OCRResponse
from ocr_service.utils.limiter import limiter

logger = logging.getLogger("ocr-service.routers.ocr")

router = APIRouter()

_doc_extractor = PersonalDocExtractor()


@router.post("/ocr", response_model=OCRResponse)
@limiter.limit("10/minute")
async def perform_ocr(
    request: Request,
    file: UploadFile = File(...),
    reconstruct: bool = False,
    advanced: bool = False,
    doc_type: str = "generic",
    _api_key: str = Depends(get_api_key),
    idempotency_key: Optional[str] = Header(default=None, alias="Idempotency-Key"),
    request_id: str = Depends(get_request_id),
    curr_settings: Settings = Depends(get_settings),
    processor: OCRProcessor = Depends(get_ocr_processor),
) -> OCRResponse:
    """
    Primary OCR entry point.
    Processes uploaded documents with optional AI-driven pixel reconstruction.
    """
    start_time = time.time()
    status = "failure"  # Default status

    try:
        # SlowAPI uses 'request' via decorator internally
        redis_client = getattr(request.app.state, "redis_client", None)

        config = ProcessingConfig(
            reconstruct=reconstruct,
            advanced=advanced,
            doc_type=doc_type,
            enable_reconstruction_config=curr_settings.enable_reconstruction,
            request_id=request_id,
            idempotency_key=idempotency_key,
            idempotency_ttl_seconds=getattr(
                curr_settings, "redis_idempotency_ttl", 3600
            ),
        )

        result = await processor.process_file(
            file=file,
            config=config,
            redis_client=redis_client,
        )
        status = "success"
        return OCRResponse(**result)
    except OCRPipelineError as e:
        OCR_ERROR_COUNT.labels(phase=e.phase, error_type=type(e).__name__).inc()
        raise  # Re-raise the exception after logging
    except Exception:
        from ocr_service.utils.tracing import (
            get_current_trace_id,  # pylint: disable=import-outside-toplevel
        )

        trace_id = get_current_trace_id()
        logger.exception(
            "Unexpected error handling OCR request | method=%s | RID=%s | TID=%s",
            request.method,
            request_id,
            trace_id,
        )
        OCR_ERROR_COUNT.labels(
            phase="request_handling", error_type="UnexpectedError"
        ).inc()
        raise  # Re-raise unexpected exceptions
    finally:
        latency = time.time() - start_time
        OCR_REQUEST_LATENCY.labels(method=request.method, status=status).observe(
            latency
        )
        OCR_REQUEST_COUNT.labels(method=request.method, status=status).inc()


@router.post("/ocr/documents", response_model=DocumentResponse)
@limiter.limit("10/minute")
async def perform_document_ocr(
    request: Request,
    file: UploadFile = File(...),
    reconstruct: bool = False,
    advanced: bool = False,
    _api_key: str = Depends(get_api_key),
    idempotency_key: Optional[str] = Header(default=None, alias="Idempotency-Key"),
    request_id: str = Depends(get_request_id),
    curr_settings: Settings = Depends(get_settings),
    processor: OCRProcessor = Depends(get_ocr_processor),
) -> DocumentResponse:
    """
    Personal document OCR endpoint.

    Processes uploaded personal documents (ID cards, passports, driver licenses,
    utility bills, bank statements, payslips, etc.) from multiple countries and
    returns structured field extraction with per-field confidence levels.

    Sensitive fields (PAN, CVV) are masked in the response and never logged.
    Low-confidence reconstructions are flagged in the ``warnings`` array.
    """
    start_time = time.time()
    status = "failure"

    try:
        redis_client = getattr(request.app.state, "redis_client", None)

        config = ProcessingConfig(
            reconstruct=reconstruct,
            advanced=advanced,
            doc_type="generic",
            enable_reconstruction_config=curr_settings.enable_reconstruction,
            request_id=request_id,
            idempotency_key=idempotency_key,
            idempotency_ttl_seconds=getattr(
                curr_settings, "redis_idempotency_ttl", 3600
            ),
        )

        result = await processor.process_file(
            file=file,
            config=config,
            redis_client=redis_client,
        )

        plain_text: str = result.get("text", "")
        document_type: str = result.get("document_type", "generic_document")
        type_confidence: float = float(result.get("type_confidence", 0.40))

        # Extract structured fields
        raw_fields, warnings = _doc_extractor.extract(plain_text, document_type)
        doc_fields = [
            DocumentField(
                name=f.name,
                value=f.value,
                raw_ocr=f.raw_ocr,
                confidence_level=f.confidence_level,
            )
            for f in raw_fields
        ]

        metadata = detect_metadata(plain_text)
        metadata["ocr_method"] = result.get("method")

        status = "success"
        return DocumentResponse(
            filename=result.get("filename", file.filename or "unknown"),
            document_type=document_type,
            type_confidence=round(type_confidence, 2),
            plain_text=plain_text,
            fields=doc_fields,
            warnings=warnings,
            metadata=metadata,
            processing_time=result.get("processing_time", round(time.time() - start_time, 3)),
            request_id=request_id,
            s3_key=result.get("s3_key"),
        )

    except OCRPipelineError as e:
        OCR_ERROR_COUNT.labels(phase=e.phase, error_type=type(e).__name__).inc()
        raise
    except Exception:
        from ocr_service.utils.tracing import (
            get_current_trace_id,  # pylint: disable=import-outside-toplevel
        )

        trace_id = get_current_trace_id()
        logger.exception(
            "Unexpected error in /ocr/documents | method=%s | RID=%s | TID=%s",
            request.method,
            request_id,
            trace_id,
        )
        OCR_ERROR_COUNT.labels(
            phase="document_ocr_handling", error_type="UnexpectedError"
        ).inc()
        raise
    finally:
        latency = time.time() - start_time
        OCR_REQUEST_LATENCY.labels(method=request.method, status=status).observe(
            latency
        )
        OCR_REQUEST_COUNT.labels(method=request.method, status=status).inc()
