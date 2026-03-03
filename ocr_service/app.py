import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from ocr_service.config import Settings, get_settings
from ocr_service.handlers import register_handlers
from ocr_service.middleware import ProcessTimeAndLoggingMiddleware
from ocr_service.routers import datasets, ocr, storage, system
from ocr_service.utils.limiter import limiter
from ocr_service.utils.monitoring import init_monitoring
from ocr_service.utils.redis_factory import (
    get_redis_client,
    verify_redis_connection,
)

logger = logging.getLogger("ocr-service")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    settings = getattr(app.state, "settings", None) or get_settings()
    app.state.redis_client = get_redis_client(settings)

    if not settings.redis_startup_check:
        redis_status = {
            "ok": True,
            "skipped": True,
            "detail": "redis_startup_check disabled",
        }
        app.state.redis_diagnostics = redis_status
        app.state.degraded = False
        logger.info("Startup dependency check skipped: %s", redis_status)
    else:
        try:
            redis_status = await verify_redis_connection(app.state.redis_client)
            app.state.redis_diagnostics = redis_status
            app.state.degraded = not redis_status.get("ok", False)
            logger.info("Startup dependency check: %s", redis_status)
        except Exception:
            logger.exception("Startup dependency check failed")
            app.state.degraded = True
            app.state.redis_diagnostics = {"ok": False}

    yield

    # Shutdown
    redis_client = getattr(app.state, "redis_client", None)
    if redis_client is not None:
        await redis_client.aclose()


def create_app(settings: Optional[Settings] = None) -> FastAPI:
    """
    Application factory for creating and configuring the FastAPI instance.
    """
    explicit_settings = settings is not None
    if settings is None:
        settings = get_settings()

    # Initialize enterprise-grade logging & monitoring
    init_monitoring(settings, release=getattr(settings, "version", None))

    app = FastAPI(
        title=settings.app_name,
        description=settings.app_description,
        version=settings.version,
        docs_url="/docs" if settings.environment != "production" else None,
        redoc_url="/redoc" if settings.environment != "production" else None,
        lifespan=lifespan,
    )

    # State
    app.state.settings = settings
    app.state.limiter = limiter

    # If explicit settings were provided (tests or custom embedding), preserve them.
    if explicit_settings:
        app.dependency_overrides[get_settings] = lambda: settings

    # Exception Handlers
    register_handlers(app)

    # Middleware
    app.add_middleware(ProcessTimeAndLoggingMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Metrics endpoint
    @app.get("/metrics")
    async def metrics():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    # Routers
    app.include_router(system.router, tags=["System"])
    app.include_router(ocr.router, tags=["OCR"])
    app.include_router(storage.router, tags=["Storage"])
    app.include_router(datasets.router, tags=["Datasets"])

    return app
