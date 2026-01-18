import time

from fastapi import APIRouter, Depends

from ocr_service.config import Settings, get_settings
from ocr_service.schemas import HealthResponse, ReconStatusResponse
from ocr_service.utils.capabilities import CapabilityProvider

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Service availability heartbeat."""
    return HealthResponse(status="healthy", timestamp=time.time())


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
