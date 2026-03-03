# pylint: disable=abstract-method,invalid-overridden-method,arguments-differ

import asyncio
import logging

import fakeredis
import pytest
import redis.exceptions

from ocr_service.config import Settings
from ocr_service.utils.redis_factory import get_redis_client, verify_redis_connection


@pytest.fixture
def dummy_settings():
    return Settings(
        ocr_api_key="fake",
        redis_host="localhost",
        redis_port=6379,
        redis_db=0,
        redis_password=None,
    )


@pytest.mark.asyncio
async def test_get_redis_client_success(dummy_settings):
    # Mock fakeredis to simulate a successful connection
    with pytest.MonkeyPatch().context() as m:
        m.setattr("redis.asyncio.Redis", fakeredis.FakeAsyncRedis)
        client = get_redis_client(dummy_settings)
        assert client is not None
        assert isinstance(client, fakeredis.FakeAsyncRedis)


@pytest.mark.asyncio
async def test_verify_redis_connection_success():
    client = fakeredis.FakeAsyncRedis()
    result = await verify_redis_connection(client)
    assert result["ok"] is True
    assert "latency_ms" in result


@pytest.mark.asyncio
async def test_verify_redis_connection_failure(caplog):
    # Simulate a Redis connection error
    class BrokenRedis:
        async def ping(self):
            raise redis.exceptions.ConnectionError("Simulated connection error")

    with caplog.at_level(logging.ERROR):
        client = BrokenRedis()
        result = await verify_redis_connection(client)
        assert result["ok"] is False
        assert "latency_ms" in result
        assert "error" in result
        assert "Simulated connection error" in result["error"]
        assert "Redis ping failed" in caplog.text


@pytest.mark.asyncio
async def test_verify_redis_connection_timeout(caplog):
    # Simulate a Redis timeout
    class SlowRedis:
        async def ping(self):
            await asyncio.sleep(2)  # Simulate a long delay
            return True

    with caplog.at_level(logging.ERROR):
        client = SlowRedis()
        result = await verify_redis_connection(client, timeout=0.1)  # Shorter timeout
        assert result["ok"] is False
        assert "latency_ms" in result
        assert "error" in result
        assert "TimeoutError" in result["error"]
        assert "Redis ping failed" in caplog.text
