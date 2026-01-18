import logging
import os

import redis.asyncio as redis

from ocr_service.config import Settings

logger = logging.getLogger("ocr-service.redis")


def get_redis_client(settings: Settings) -> redis.Redis:
    """
    Factory for creating an asynchronous Redis client based on settings.
    """
    host = os.getenv(
        "REDIS_HOST",
        settings.redis_host if hasattr(settings, "redis_host") else "localhost",
    )
    port = int(
        os.getenv(
            "REDIS_PORT",
            settings.redis_port if hasattr(settings, "redis_port") else 6379,
        )
    )
    db = int(
        os.getenv("REDIS_DB", settings.redis_db if hasattr(settings, "redis_db") else 0)
    )

    logger.info(
        "Initializing Redis client | Host: %s | Port: %d | DB: %d", host, port, db
    )

    return redis.Redis(
        host=host,
        port=port,
        db=db,
        decode_responses=False,
    )
