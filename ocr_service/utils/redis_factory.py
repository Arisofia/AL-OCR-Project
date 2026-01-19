import logging
import os

import redis.asyncio as redis

from ocr_service.config import Settings

logger = logging.getLogger("ocr-service.redis")


def get_redis_client(settings: Settings) -> redis.Redis:
    """
    Factory for creating an asynchronous Redis client based on settings.
    """
    host = os.getenv("REDIS_HOST", settings.redis_host)
    port = int(os.getenv("REDIS_PORT", str(settings.redis_port)))
    db = int(os.getenv("REDIS_DB", str(settings.redis_db)))
    password = os.getenv("REDIS_PASSWORD", settings.redis_password)

    logger.info(
        "Initializing Redis client | Host: %s | Port: %d | DB: %d", host, port, db
    )

    return redis.Redis(
        host=host,
        port=port,
        db=db,
        password=password,
        decode_responses=False,
    )
