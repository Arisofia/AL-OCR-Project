"""
This module contains the routers for the system endpoints.
"""

import logging

from fastapi import APIRouter, Depends

from ocr_service.config import Settings, get_settings
from ocr_service.schemas import ReconStatusResponse
from ocr_service.utils.capabilities import CapabilityProvider

logger = logging.getLogger("ocr-service.routers.system")

router = APIRouter()


@router.get("/reconstruction/status", response_model=ReconStatusResponse)
async def get_reconstruction_status(
    curr_settings: Settings = Depends(get_settings),
) -> ReconStatusResponse:
    """
    Get the current status of reconstruction capabilities.
    """
    return ReconStatusResponse(
        reconstruction_enabled=curr_settings.enable_reconstruction,
        package_installed=CapabilityProvider.is_reconstruction_available(),
        package_version=CapabilityProvider.get_reconstruction_version(),
    )