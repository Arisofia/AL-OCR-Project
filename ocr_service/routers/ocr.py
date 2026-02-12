import logging
import time

from fastapi import APIRouter, Depends, File, Request, UploadFile

from ocr_service.config import Settings, get_settings
from ocr_service.exceptions import OCRPipelineError
from ocr_service.metrics import OCR_ERROR_COUNT, OCR_REQUEST_COUNT, OCR_REQUEST_LATENCY
from ocr_service.modules.processor import OCRProcessor
from ocr_service.routers.deps import get_api_key, get_ocr_processor, get_request_id
from ocr_service.schemas import OCRResponse
from ocr_service.utils.limiter import limiter

logger = logging.getLogger("ocr-service.routers.ocr")

router = APIRouter()


@router.post("/ocr", response_model=OCRResponse)
@limiter.limit("10/minute")
async def perform_ocr(
    request: Request,
    file: UploadFile = File(...),
    reconstruct: bool = False,
    advanced: bool = False,
    doc_type: str = "generic",
    _api_key: str = Depends(get_api_key),
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
        result = await processor.process_file(
            file=file,
            reconstruct=reconstruct,
            advanced=advanced,
            doc_type=doc_type,
            enable_reconstruction_config=curr_settings.enable_reconstruction,
            request_id=request_id,
        )
        status = "success"
        return OCRResponse(**result)
    except OCRPipelineError as e:
        OCR_ERROR_COUNT.labels(phase=e.phase, error_type=type(e).__name__).inc()
        raise  # Re-raise the exception after logging
    except Exception:
        from ocr_service.utils.tracing import get_current_trace_id

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
