import logging
from typing import Any, Optional

import sentry_sdk

from .custom_logging import setup_logging

# OpenTelemetry imports
try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
except ImportError:
    trace = None


def init_monitoring(settings: Any, integrations: Optional[list[Any]] = None, **kwargs):
    """
    Unified initialization for logging and Sentry monitoring.

    Args:
        settings: Application settings object containing sentry_dsn and environment.
        integrations: Optional list of Sentry integrations to add.
        **kwargs: Additional parameters for sentry_sdk.init (e.g., release).
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
            **kwargs,
        )
        logger.info(
            "Sentry SDK initialized with %d integrations", len(default_integrations)
        )
    else:
        logger.info("Sentry DSN not configured, skipping SDK initialization")

    # 3. Initialize OpenTelemetry
    if trace and not isinstance(trace.get_tracer_provider(), TracerProvider):
        tracer_provider = TracerProvider()
        trace.set_tracer_provider(tracer_provider)
        tracer_provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
        logger.info("OpenTelemetry SDK initialized with ConsoleSpanExporter")
