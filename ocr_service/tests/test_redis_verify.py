"""Unit tests for Redis connection verification helper."""

import asyncio

import pytest

from ocr_service.utils.redis_factory import verify_redis_connection


@pytest.mark.asyncio
async def test_verify_redis_success():
    class DummyClient:
        """Redis client stub returning successful ping."""

        async def ping(self):
            await asyncio.sleep(0)
            return True

    res = await verify_redis_connection(DummyClient())
    assert res["ok"] is True
    assert "latency_ms" in res


@pytest.mark.asyncio
async def test_verify_redis_failure():
    class DummyClient:
        """Redis client stub that raises on ping."""

        async def ping(self):
            await asyncio.sleep(0)
            raise RuntimeError("no conn")

    res = await verify_redis_connection(DummyClient())
    assert res["ok"] is False
    assert "error" in res
