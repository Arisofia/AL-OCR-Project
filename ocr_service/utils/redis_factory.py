"""
Redis client factory for OCR service.

This module provides helpers to initialize and validate Redis connectivity
using environment variables and application settings.
"""

import logging
import os

import redis.asyncio as redis

from ocr_service.config import Settings

logger = logging.getLogger("ocr-service.redis")


class RedisInitializationError(RuntimeError):
    """Raised when Redis initialization or validation fails."""


def _env_or_default(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value not in (None, "") else default


def get_redis_client(settings: Settings) -> redis.Redis:
    """Create an asynchronous Redis client from settings and environment."""
    host = _env_or_default("REDIS_HOST", settings.redis_host)
    port = int(_env_or_default("REDIS_PORT", str(settings.redis_port)))
    db = int(_env_or_default("REDIS_DB", str(settings.redis_db)))
    password = os.getenv("REDIS_PASSWORD", settings.redis_password)

    logger.info(
        "Initializing Redis client | host=%s | port=%d | db=%d",
        host,
        port,
        db,
    )

    return redis.Redis(
        host=host,
        port=port,
        db=db,
        password=password,
        decode_responses=False,
        socket_connect_timeout=3,
        socket_timeout=3,
        health_check_interval=30,
        retry_on_timeout=True,
    )


async def verify_redis_connection(client: redis.Redis, settings: Settings) -> None:
    """Perform startup ping check for Redis connectivity."""
    if not settings.redis_startup_check:
        logger.info("Redis startup check disabled by configuration")
        return

    try:
        await client.ping()
        logger.info("Redis connectivity check succeeded")
    except Exception as exc:  # pragma: no cover - covered through API lifecycle tests
        logger.exception("Redis connectivity check failed")
        raise RedisInitializationError("Redis startup connectivity check failed") from exc
