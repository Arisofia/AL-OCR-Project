"""Tests for Redis client factory and connectivity diagnostics."""


import asyncio
import logging
from typing import Any, cast

import fakeredis
import pytest
import redis.exceptions

from ocr_service.config import Settings
from ocr_service.utils.redis_factory import get_redis_client, verify_redis_connection


@pytest.fixture(name="dummy_settings")
def fixture_dummy_settings() -> Settings:
    """Return minimal settings required to initialize the Redis client."""
    return Settings(
        ocr_api_key="fake",
        redis_host="localhost",
        redis_port=6379,
        redis_db=0,
        redis_password=None,
    )


@pytest.mark.asyncio
async def test_get_redis_client_success(dummy_settings):
    """`get_redis_client` should return a redis client object."""
    with pytest.MonkeyPatch().context() as m:
        m.setattr("redis.asyncio.Redis", fakeredis.FakeAsyncRedis)
        client = get_redis_client(dummy_settings)
        assert client is not None
        assert isinstance(client, fakeredis.FakeAsyncRedis)


@pytest.mark.asyncio
async def test_verify_redis_connection_success():
    """`verify_redis_connection` should report success for healthy client."""
    client = fakeredis.FakeAsyncRedis()
    result = await verify_redis_connection(client)
    assert result["ok"] is True
    assert "latency_ms" in result


@pytest.mark.asyncio
async def test_verify_redis_connection_failure(caplog):
    """Connection errors should be reported with `ok=False` and error detail."""
    class BrokenRedis:
        """Redis test double that always raises on ping."""

        async def ping(self):
            """Raise a connection error to emulate an unavailable Redis server."""
            raise redis.exceptions.ConnectionError("Simulated connection error")

    with caplog.at_level(logging.ERROR):
        client = BrokenRedis()
        result = await verify_redis_connection(cast(Any, client))
        assert result["ok"] is False
        assert "latency_ms" in result
        assert "error" in result
        assert "Simulated connection error" in result["error"]
        assert "Redis ping failed" in caplog.text


@pytest.mark.asyncio
async def test_verify_redis_connection_timeout(caplog):
    """Timeouts should be reported with `ok=False` and timeout error detail."""
    class SlowRedis:
        """Redis test double that responds too slowly."""

        async def ping(self):
            """Sleep long enough to trigger asyncio timeout in the verifier."""
            await asyncio.sleep(2)
            return True

    with caplog.at_level(logging.ERROR):
        client = SlowRedis()
        result = await verify_redis_connection(
            cast(Any, client), timeout=0.1
        )
        assert result["ok"] is False
        assert "latency_ms" in result
        assert "error" in result
        assert "TimeoutError" in result["error"]
        assert "Redis ping failed" in caplog.text
