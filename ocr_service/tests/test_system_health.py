import pytest

from ocr_service.routers.system import health_check


@pytest.mark.asyncio
async def test_health_degraded(monkeypatch):
    async def fake_verify(_client, _timeout=1.0):
        return {"ok": False, "error": "no"}

    monkeypatch.setattr(
        "ocr_service.utils.redis_factory.verify_redis_connection",
        fake_verify,
    )

    res = await health_check()
    assert res.status == "degraded"
    assert res.components["redis"]["ok"] is False
