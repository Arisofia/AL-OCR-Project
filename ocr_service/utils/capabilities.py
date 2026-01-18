import logging

logger = logging.getLogger("ocr-service.capabilities")


class CapabilityProvider:
    """
    Central registry for optional system capabilities and package availability.
    """

    _RECON_AVAILABLE: bool = False
    _RECON_VERSION: str = "not-installed"
    _INITIALIZED: bool = False

    @classmethod
    def initialize(cls):
        """Detects optional packages at runtime."""
        if cls._INITIALIZED:
            return

        try:
            import ocr_reconstruct

            cls._RECON_AVAILABLE = True
            cls._RECON_VERSION = getattr(ocr_reconstruct, "__version__", "unknown")
            logger.info(
                "OCR Reconstruction capability detected (version: %s)",
                cls._RECON_VERSION,
            )
        except ImportError:
            cls._RECON_AVAILABLE = False
            cls._RECON_VERSION = "not-installed"
            logger.info("OCR Reconstruction capability not available")
        except Exception as e:
            cls._RECON_AVAILABLE = False
            cls._RECON_VERSION = f"error: {e!s}"
            logger.warning("Error detecting OCR Reconstruction: %s", e)

        cls._INITIALIZED = True

    @classmethod
    def is_reconstruction_available(cls) -> bool:
        cls.initialize()
        return cls._RECON_AVAILABLE

    @classmethod
    def get_reconstruction_version(cls) -> str:
        cls.initialize()
        return cls._RECON_VERSION
