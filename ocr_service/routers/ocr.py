import logging
import time
from typing import Optional

from fastapi import APIRouter, Depends, File, Header, Request, UploadFile

from ocr_service.config import Settings, get_settings
from ocr_service.exceptions import OCRPipelineError
from ocr_service.metrics import OCR_ERROR_COUNT, OCR_REQUEST_COUNT, OCR_REQUEST_LATENCY
from ocr_service.modules.decision_readiness import (
    MANDATORY_FIELDS,
    compute_decision_readiness,
    quality_band,
)
from ocr_service.modules.personal_doc_extractor import (
    PersonalDocExtractor,
    detect_metadata,
)
from ocr_service.modules.processor import OCRProcessor, ProcessingConfig
from ocr_service.routers.deps import get_api_key, get_ocr_processor, get_request_id
from ocr_service.schemas import (
    DocumentAnalytics,
    DocumentField,
    DocumentResponse,
    OCRResponse,
)
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
    doc_type: str = "generic",
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

    Example response::

        {
            "filename": "passport.jpg",
            "document_type": "passport",
            "type_confidence": 0.80,
            "plain_text": "PASSPORT\\nSurname: SMITH...",
            "fields": [
                {"name": "document_number", "value": "AB123456",
                 "raw_ocr": "AB123456", "confidence_level": "high"},
                {"name": "date_of_birth", "value": "15-03-1985",
                 "raw_ocr": "15/03/1985", "confidence_level": "medium"}
            ],
            "warnings": [
                "date_of_birth partially reconstructed from OCR output; verify manually"
            ],
            "metadata": {"language_guess": "en", "country_guess": null,
                         "ocr_method": "gemini"},
            "processing_time": 1.23,
            "request_id": "abc-123",   // optional – null when not set
            "s3_key": null             // optional – null when no S3 upload
        }
    """
    start_time = time.time()
    status = "failure"

    try:
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

        plain_text: str = result.get("text", "")
        document_type: str = result.get("document_type", "generic_document")
        default_type_confidence: float = 0.40
        raw_type_confidence = result.get("type_confidence")
        if raw_type_confidence is None:
            type_confidence: float = default_type_confidence
        else:
            try:
                type_confidence = float(raw_type_confidence)
            except (TypeError, ValueError):
                type_confidence = default_type_confidence

        _low_confidence_threshold = 0.65
        if (
            doc_type != "generic"
            and type_confidence < _low_confidence_threshold
            and document_type in {"generic_document", "statement", "form"}
        ):
            document_type = doc_type
            type_confidence = _low_confidence_threshold

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

        # --- Analytics -------------------------------------------------------
        decision_readiness = compute_decision_readiness(
            document_type, raw_fields, type_confidence
        )

        band = quality_band(type_confidence)
        requires_review = band in ("fair", "poor") or not decision_readiness["ready"]

        remediation_hints: list[str] = []
        if band == "poor":
            remediation_hints.append(
                "Image quality is poor; consider re-scanning at higher resolution."
            )
        elif band == "fair":
            remediation_hints.append(
                "Image quality is fair; manual verification recommended."
            )
        for mf in decision_readiness.get("missing_mandatory", []):
            remediation_hints.append(
                f"Mandatory field '{mf}' could not be extracted; verify manually."
            )

        expected_count = len(MANDATORY_FIELDS.get(document_type, []))
        # Total number of fields extracted (mandatory + optional)
        extracted_count = len(doc_fields)
        # Number of mandatory fields that are present
        missing_mandatory = decision_readiness.get("missing_mandatory", []) or []
        mandatory_present = max(0, expected_count - len(missing_mandatory))
        if expected_count:
            ratio = mandatory_present / expected_count
            # Clamp ratio to [0, 1] to avoid over- or underflow due to inconsistencies
            ratio = max(0.0, min(1.0, ratio))
            completeness = round(ratio, 4)
        else:
            completeness = None

        analytics = DocumentAnalytics(
            decision_readiness=decision_readiness,
            quality_band=band,
            requires_manual_review=requires_review,
            remediation_hints=remediation_hints,
            field_completeness_ratio=completeness,
            fields_extracted_count=extracted_count,
            fields_expected_count=expected_count,
        )
        # --- End analytics ---------------------------------------------------

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
            analytics=analytics,
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
