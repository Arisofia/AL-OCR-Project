from typing import Optional, List, Any
import logging
import sentry_sdk
from .custom_logging import setup_logging


def init_monitoring(settings: Any, integrations: Optional[List[Any]] = None):
    """
    Unified initialization for logging and Sentry monitoring.

    Args:
        settings: Application settings object containing sentry_dsn and environment.
        integrations: Optional list of Sentry integrations to add.
    """
    # 1. Initialize logging
    log_level = logging.INFO if settings.environment == "production" else logging.DEBUG
    setup_logging(level=log_level)
    logger = logging.getLogger("ocr-service.init")

    # 2. Initialize Sentry
    if settings.sentry_dsn:
        default_integrations = []
        if integrations:
            default_integrations.extend(integrations)

        sentry_sdk.init(
            dsn=settings.sentry_dsn,
            environment=settings.environment,
            integrations=default_integrations,
            traces_sample_rate=1.0 if settings.environment != "production" else 0.1,
        )
        logger.info(
            "Sentry SDK initialized with %d integrations", len(default_integrations)
        )
    else:
        logger.info("Sentry DSN not configured, skipping SDK initialization")
