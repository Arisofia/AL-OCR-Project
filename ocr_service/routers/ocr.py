"""OCR API routes for generic and document-aware extraction endpoints."""

import logging
import time
from typing import Annotated, Literal, Optional

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
    ExtractedField,
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
from ocr_service.utils.tracing import get_current_trace_id

logger = logging.getLogger("ocr-service.routers.ocr")

router = APIRouter()

_doc_extractor = PersonalDocExtractor()


def _normalize_confidence_level(value: str) -> Literal["high", "medium", "low"]:
    normalized = (value or "").strip().lower()
    confidence_map: dict[str, Literal["high", "medium", "low"]] = {
        "high": "high",
        "medium": "medium",
        "low": "low",
    }
    return confidence_map.get(normalized, "low")


def _coerce_optional_float(value: object) -> Optional[float]:
    if value is None:
        return None
    if not isinstance(value, (int, float, str)):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_type_confidence(raw_type_confidence: object, default_value: float) -> float:
    if raw_type_confidence is None:
        return default_value
    if not isinstance(raw_type_confidence, (int, float, str)):
        return default_value
    try:
        return float(raw_type_confidence)
    except (TypeError, ValueError):
        return default_value


def _coerce_str(value: object, default_value: str) -> str:
    return value if isinstance(value, str) else default_value


def _coerce_processing_time(value: object, default_value: float) -> float:
    coerced = _coerce_optional_float(value)
    return coerced if coerced is not None else default_value


def _resolve_document_type(
    requested_doc_type: str,
    detected_document_type: str,
    type_confidence: float,
    low_confidence_threshold: float,
) -> tuple[str, float]:
    if (
        requested_doc_type != "generic"
        and type_confidence < low_confidence_threshold
        and detected_document_type in {"generic_document", "statement", "form"}
    ):
        return requested_doc_type, low_confidence_threshold
    return detected_document_type, type_confidence


def _build_document_fields(raw_fields: list[ExtractedField]) -> list[DocumentField]:
    return [
        DocumentField(
            name=field.name,
            value=field.value,
            raw_ocr=field.raw_ocr,
            confidence_level=_normalize_confidence_level(field.confidence_level),
        )
        for field in raw_fields
    ]


def _build_remediation_hints(
    band: str,
    missing_mandatory_fields: list[str],
) -> list[str]:
    hints: list[str] = []
    if band == "poor":
        hints.append(
            "Image quality is poor; consider re-scanning at higher resolution."
        )
    elif band == "fair":
        hints.append(
            "Image quality is fair; manual verification recommended."
        )
    hints.extend(
        [
            (
                f"Mandatory field '{mandatory_field}' could not be extracted; "
                "verify manually."
            )
            for mandatory_field in missing_mandatory_fields
        ]
    )
    return hints


def _compute_completeness_ratio(expected_count: int, missing_mandatory: list[str]) -> Optional[float]:
    mandatory_present = max(0, expected_count - len(missing_mandatory))
    if not expected_count:
        return None
    ratio = mandatory_present / expected_count
    ratio = max(0.0, min(1.0, ratio))
    return round(ratio, 4)


def _build_document_analytics(
    result: dict[str, object],
    document_type: str,
    raw_fields: list[ExtractedField],
    doc_fields: list[DocumentField],
    type_confidence: float,
) -> DocumentAnalytics:
    decision_readiness = compute_decision_readiness(
        document_type, raw_fields, type_confidence
    )
    band = quality_band(type_confidence)
    requires_review = band in ("fair", "poor") or not decision_readiness["ready"]

    missing_mandatory = decision_readiness.get("missing_mandatory", []) or []
    expected_count = len(MANDATORY_FIELDS.get(document_type, []))
    extracted_count = len(doc_fields)
    remediation_hints = _build_remediation_hints(band, missing_mandatory)
    completeness = _compute_completeness_ratio(expected_count, missing_mandatory)

    return DocumentAnalytics(
        pixel_coverage_ratio=_coerce_optional_float(result.get("pixel_coverage_ratio")),
        readability_index=_coerce_optional_float(result.get("readability_index")),
        decision_readiness=decision_readiness,
        iteration_convergence=_coerce_optional_float(result.get("iteration_convergence")),
        pixel_rescue_applied=bool(result.get("pixel_rescue_applied", False)),
        quality_band=band,
        requires_manual_review=requires_review,
        remediation_hints=remediation_hints,
        field_completeness_ratio=completeness,
        fields_extracted_count=extracted_count,
        fields_expected_count=expected_count,
    )


def _build_document_response(
    *,
    result: dict[str, object],
    file: UploadFile,
    doc_type: str,
    request_id: str,
    start_time: float,
) -> DocumentResponse:
    plain_text = _coerce_str(result.get("text"), "")
    detected_document_type = _coerce_str(
        result.get("document_type"), "generic_document"
    )
    default_type_confidence: float = 0.40
    type_confidence = _coerce_type_confidence(
        result.get("type_confidence"), default_type_confidence
    )
    _low_confidence_threshold = 0.65
    document_type, type_confidence = _resolve_document_type(
        doc_type,
        detected_document_type,
        type_confidence,
        _low_confidence_threshold,
    )

    raw_fields, warnings = _doc_extractor.extract(plain_text, document_type)
    doc_fields = _build_document_fields(raw_fields)

    metadata = detect_metadata(plain_text)
    metadata["ocr_method"] = result.get("method")
    analytics = _build_document_analytics(
        result=result,
        document_type=document_type,
        raw_fields=raw_fields,
        doc_fields=doc_fields,
        type_confidence=type_confidence,
    )

    return DocumentResponse(
        filename=_coerce_str(result.get("filename"), file.filename or "unknown"),
        document_type=document_type,
        type_confidence=round(type_confidence, 2),
        plain_text=plain_text,
        fields=doc_fields,
        warnings=warnings,
        metadata=metadata,
        processing_time=_coerce_processing_time(
            result.get("processing_time"), round(time.time() - start_time, 3)
        ),
        request_id=request_id,
        s3_key=_coerce_str(result.get("s3_key"), "") or None,
        analytics=analytics,
    )


@router.post("/ocr")
@limiter.limit("10/minute")
async def perform_ocr(
    request: Request,
    file: Annotated[UploadFile, File(...)],
    _api_key: Annotated[str, Depends(get_api_key)],
    request_id: Annotated[str, Depends(get_request_id)],
    curr_settings: Annotated[Settings, Depends(get_settings)],
    processor: Annotated[OCRProcessor, Depends(get_ocr_processor)],
    reconstruct: bool = False,
    advanced: bool = False,
    doc_type: str = "generic",
    idempotency_key: Annotated[
        Optional[str], Header(alias="Idempotency-Key")
    ] = None,
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


@router.post("/ocr/documents")
@limiter.limit("10/minute")
async def perform_document_ocr(
    request: Request,
    file: Annotated[UploadFile, File(...)],
    _api_key: Annotated[str, Depends(get_api_key)],
    request_id: Annotated[str, Depends(get_request_id)],
    curr_settings: Annotated[Settings, Depends(get_settings)],
    processor: Annotated[OCRProcessor, Depends(get_ocr_processor)],
    reconstruct: bool = False,
    advanced: bool = False,
    doc_type: str = "generic",
    idempotency_key: Annotated[
        Optional[str], Header(alias="Idempotency-Key")
    ] = None,
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

        status = "success"
        return _build_document_response(
            result=result,
            file=file,
            doc_type=doc_type,
            request_id=request_id,
            start_time=start_time,
        )

    except OCRPipelineError as e:
        OCR_ERROR_COUNT.labels(phase=e.phase, error_type=type(e).__name__).inc()
        raise
    except Exception:
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
