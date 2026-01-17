from fastapi import APIRouter, Depends, HTTPException, Request

from ocr_service.routers.deps import get_api_key, get_storage_service
from ocr_service.schemas import PresignRequest, PresignResponse
from ocr_service.services.storage import StorageService
from ocr_service.utils.limiter import limiter

router = APIRouter()


@router.post("/presign", response_model=PresignResponse)
@limiter.limit("5/minute")
async def generate_presigned_post(
    request: Request,  # noqa: ARG001
    req: PresignRequest,
    _api_key: str = Depends(get_api_key),
    storage: StorageService = Depends(get_storage_service),
) -> PresignResponse:
    """
    Generates a secure, time-limited S3 POST URL for client-side direct uploads.
    """
    if not storage.bucket_name:
        raise HTTPException(status_code=500, detail="S3 bucket not configured")

    try:
        post = storage.generate_presigned_post(
            key=req.key,
            content_type=req.content_type,
            expires_in=req.expires_in,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail="Could not generate presigned post"
        ) from exc

    return PresignResponse(url=post["url"], fields=post["fields"])
