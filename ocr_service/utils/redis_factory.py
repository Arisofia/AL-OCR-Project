"""
Redis client factory for OCR service.

This module provides helpers to initialize and validate Redis connectivity
using environment variables and application settings.
"""

import logging

import redis.asyncio as redis

from ocr_service.config import Settings

logger = logging.getLogger("ocr-service.redis")


class RedisInitializationError(RuntimeError):
    """Raised when Redis initialization or validation fails."""


def get_redis_client(settings: Settings) -> redis.Redis:
    """
    Factory for creating an asynchronous Redis client based on settings.

    This function takes application settings as input and returns an
    initialized Redis client.
    """
    host = settings.redis_host
    port = settings.redis_port
    db = settings.redis_db
    password = settings.redis_password

    logger.info(
        "Initializing Redis client | host=%s | port=%d | db=%d",
        host,
        port,
        db,
    )

    # Create the Redis client instance with conservative timeouts to
    # avoid long blocking calls in high-concurrency environments.
    return redis.Redis(
        host=host,
        port=port,
        db=db,
        password=password,
        decode_responses=False,
        socket_connect_timeout=1.0,
        socket_timeout=1.0,
    )


async def verify_redis_connection(client: redis.Redis, timeout: float = 1.0) -> dict:
    """Verify that Redis is reachable and return diagnostics.

    Returns a dict: {"ok": bool, "latency_ms": float|None, "error": Optional[str]}
    """
    import time as _time

    start = _time.time()
    try:
        # Bound the ping with asyncio.wait_for to avoid long hangs
        import asyncio

        await asyncio.wait_for(client.ping(), timeout=timeout)
        latency = round(((_time.time() - start) * 1000), 2)
        return {"ok": True, "latency_ms": latency}
    except Exception as e:  # pragma: no cover - defensive
        latency = round(((_time.time() - start) * 1000), 2)
        logger.exception("Redis ping failed: %s", e)
        error_detail = f"{type(e).__name__}: {e}" if str(e) else type(e).__name__
        return {"ok": False, "latency_ms": latency, "error": error_detail}
