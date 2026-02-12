from fastapi import APIRouter, Depends, File, Header, Request, UploadFile

from ocr_service.config import Settings, get_settings
from ocr_service.modules.processor import OCRProcessor
from ocr_service.routers.deps import get_api_key, get_ocr_processor, get_request_id
from ocr_service.schemas import OCRResponse
from ocr_service.utils.limiter import limiter

router = APIRouter()


@router.post("/ocr", response_model=OCRResponse)
@limiter.limit("10/minute")
async def perform_ocr(
    request: Request,  # noqa: ARG001
    file: UploadFile = File(...),
    reconstruct: bool = False,
    advanced: bool = False,
    doc_type: str = "generic",
    _api_key: str = Depends(get_api_key),
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    request_id: str = Depends(get_request_id),
    curr_settings: Settings = Depends(get_settings),
    processor: OCRProcessor = Depends(get_ocr_processor),
) -> OCRResponse:
    """
    Primary OCR entry point.
    Processes uploaded documents with optional AI-driven pixel reconstruction.
    """
    # SlowAPI uses 'request' via decorator internally
    redis_client = getattr(request.app.state, "redis_client", None)

    result = await processor.process_file(
        file=file,
        reconstruct=reconstruct,
        advanced=advanced,
        doc_type=doc_type,
        enable_reconstruction_config=curr_settings.enable_reconstruction,
        request_id=request_id,
        redis_client=redis_client,
        idempotency_key=idempotency_key,
    )

    return OCRResponse(**result)
