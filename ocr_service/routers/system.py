import time

from fastapi import APIRouter, Depends

from ocr_service.config import Settings, get_settings
from ocr_service.schemas import HealthResponse, ReconStatusResponse

# Runtime package detection for optional reconstruction capability
try:
    import ocr_reconstruct as _ocr_reconstruct_pkg  # type: ignore

    RECON_PKG_AVAILABLE = True
    RECON_PKG_VERSION = getattr(_ocr_reconstruct_pkg, "__version__", "unknown")
except Exception:
    RECON_PKG_AVAILABLE = False
    RECON_PKG_VERSION = "not-installed"

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
        package_installed=RECON_PKG_AVAILABLE,
        package_version=RECON_PKG_VERSION,
    )
