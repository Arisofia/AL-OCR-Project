"""System router for health and reconstruction capability endpoints."""

import inspect
import logging
import time
from typing import Any

from fastapi import APIRouter, Depends

from ocr_service.config import Settings, get_settings
from ocr_service.schemas import HealthResponse, ReconStatusResponse
from ocr_service.services.storage import StorageService
from ocr_service.utils import redis_factory
from ocr_service.utils.capabilities import CapabilityProvider

logger = logging.getLogger("ocr-service.routers.system")
router = APIRouter()


# Wrappers keep health checks patchable in tests while remaining runtime-simple.
def get_redis_client(settings: Settings):
    return redis_factory.get_redis_client(settings)


async def verify_redis_connection(client: Any, timeout: float = 1.0):
    result = redis_factory.verify_redis_connection(client, timeout=timeout)
    if inspect.isawaitable(result):
        return await result
    return result


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Service availability heartbeat with component checks."""
    curr_settings = get_settings()
    components: dict[str, Any] = {}

    # Redis check
    try:
        redis_client = get_redis_client(curr_settings)
        redis_res = verify_redis_connection(redis_client)
        if inspect.isawaitable(redis_res):
            redis_res = await redis_res
        components["redis"] = redis_res
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Health check redis dependency failed")
        components["redis"] = {"ok": False, "error": str(exc)}

    # Storage (S3) check
    try:
        storage_service = StorageService()
        components["s3"] = {"ok": storage_service.check_connection()}
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Health check storage dependency failed")
        components["s3"] = {"ok": False, "error": str(exc)}

    overall_ok = all(component.get("ok") for component in components.values())
    status = "ok" if overall_ok else "degraded"

    return HealthResponse(status=status, timestamp=time.time(), components=components)


@router.get("/recon/status", response_model=ReconStatusResponse)
async def recon_status(
    curr_settings: Settings = Depends(get_settings),
) -> ReconStatusResponse:
    """Retrieves pixel reconstruction capability and version metadata."""
    return ReconStatusResponse(
        reconstruction_enabled=curr_settings.enable_reconstruction,
        package_installed=CapabilityProvider.is_reconstruction_available(),
        package_version=CapabilityProvider.get_reconstruction_version(),
    )
