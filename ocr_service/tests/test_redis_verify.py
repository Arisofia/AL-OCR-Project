import pytest

from ocr_service.utils.redis_factory import verify_redis_connection


@pytest.mark.asyncio
async def test_verify_redis_success():
    class DummyClient:
        async def ping(self):
            return True

    res = await verify_redis_connection(DummyClient())  # type: ignore[arg-type]
    assert res["ok"] is True
    assert "latency_ms" in res


@pytest.mark.asyncio
async def test_verify_redis_failure():
    class DummyClient:
        async def ping(self):
            raise RuntimeError("no conn")

    res = await verify_redis_connection(DummyClient())  # type: ignore[arg-type]
    assert res["ok"] is False
    assert "error" in res
