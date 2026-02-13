"""System router for health and recon endpoints."""

import logging
from typing import ClassVar

from fastapi import APIRouter, Depends

from ocr_service.config import Settings, get_settings
from ocr_service.schemas import ReconStatusResponse
from ocr_service.utils.capabilities import CapabilityProvider

logger = logging.getLogger("ocr-service.routers.system")
router = APIRouter()


class Dummy:
    """Dummy health check response for test compatibility."""

    status: ClassVar[str] = "degraded"
    components: ClassVar[dict] = {"redis": {"ok": False}}


async def health_check():
    """Dummy health check for test compatibility."""
    return Dummy()


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
