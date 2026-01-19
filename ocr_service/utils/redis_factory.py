"""
Redis client factory for OCR service.

This module provides a function to initialize and configure a Redis client
using environment variables and application settings.
"""

import logging
import os

import redis.asyncio as redis

from ocr_service.config import Settings

logger = logging.getLogger("ocr-service.redis")


def get_redis_client(settings: Settings) -> redis.Redis:
    """
    Factory for creating an asynchronous Redis client based on settings.

    This function takes application settings as input and returns an
    initialized Redis client. It uses environment variables as a fallback
    for the Redis host, port, database index, and password, if they are
    not provided in the application settings.

    The returned Redis client is configured with the specified host,
    port, database index, and password. The decode_responses parameter
    is set to False to ensure that Redis responses are not decoded as
    UTF-8 strings.

    Parameters:
        settings (Settings): The application settings.

    Returns:
        redis.Redis: An asynchronous Redis client.
    """
    host = os.getenv("REDIS_HOST", settings.redis_host)
    port = int(os.getenv("REDIS_PORT", str(settings.redis_port)))
    db = int(os.getenv("REDIS_DB", str(settings.redis_db)))
    password = os.getenv("REDIS_PASSWORD", settings.redis_password)

    log_message = f"Initializing Redis client | Host: {host} | Port: {port} | DB: {db}"
    logger.info(log_message)

    # Create the Redis client instance
    return redis.Redis(
        host=host,
        port=port,
        db=db,
        password=password,
        decode_responses=False,
    )
