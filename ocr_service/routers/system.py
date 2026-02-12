import time

from fastapi import APIRouter, Depends

from ocr_service.config import Settings, get_settings
from ocr_service.schemas import HealthResponse, ReconStatusResponse
from ocr_service.utils.capabilities import CapabilityProvider

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Service availability heartbeat with component checks."""
    from ocr_service.services.storage import StorageService
    from ocr_service.utils.redis_factory import (
        get_redis_client,
        verify_redis_connection,
    )

    curr_settings = get_settings()

    components: dict = {}

    # Redis check
    try:
        redis_client = get_redis_client(curr_settings)
        redis_res = await verify_redis_connection(redis_client)
        components["redis"] = redis_res
    except Exception as e:  # pragma: no cover - defensive
        components["redis"] = {"ok": False, "error": str(e)}

    # Storage (S3) check
    try:
        storage_service = StorageService(
            bucket_name=getattr(curr_settings, "s3_bucket_name", None)
        )
        components["s3"] = {"ok": storage_service.check_connection()}
    except Exception as e:  # pragma: no cover - defensive
        components["s3"] = {"ok": False, "error": str(e)}

    overall_ok = all(c.get("ok") for c in components.values())
    status = "ok" if overall_ok else "degraded"

    return HealthResponse(status=status, timestamp=time.time(), components=components)


@router.get("/recon/status", response_model=ReconStatusResponse)
async def recon_status(
    curr_settings: Settings = Depends(get_settings),
) -> ReconStatusResponse:
    """Retrieves pixel reconstruction capability and versioning metadata."""
    return ReconStatusResponse(
        reconstruction_enabled=curr_settings.enable_reconstruction,
        package_installed=CapabilityProvider.is_reconstruction_available(),
        package_version=CapabilityProvider.get_reconstruction_version(),
    )
