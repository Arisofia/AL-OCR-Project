"""Monitoring utilities: Sentry integration and OpenTelemetry tracing setup."""

import logging
import os
from typing import Any, Optional, cast

import sentry_sdk

from .custom_logging import setup_logging

trace = None
TracerProvider = None
BatchSpanProcessor = None
ConsoleSpanExporter = None

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
    log_level = logging.INFO if settings.environment == "production" else logging.DEBUG
    setup_logging(level=log_level)
    logger = logging.getLogger("ocr-service.init")

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

    if settings.environment == "test" or os.getenv("PYTEST_CURRENT_TEST"):
        logger.debug("Skipping OpenTelemetry console exporter in test environment")
        return

    if (
        trace is not None
        and TracerProvider is not None
        and BatchSpanProcessor is not None
        and ConsoleSpanExporter is not None
    ):
        trace_api = cast(Any, trace)
        tracer_provider_cls = cast(type[Any], TracerProvider)
        if not isinstance(trace_api.get_tracer_provider(), tracer_provider_cls):
            tracer_provider = tracer_provider_cls()
            trace_api.set_tracer_provider(tracer_provider)
            tracer_provider.add_span_processor(
                cast(Any, BatchSpanProcessor)(cast(Any, ConsoleSpanExporter)())
            )
            logger.info("OpenTelemetry SDK initialized with ConsoleSpanExporter")
