import pytest

from ocr_service.config import Settings
from ocr_service.routers.system import health_check


@pytest.mark.asyncio
async def test_health_degraded(monkeypatch):
    settings = Settings(ocr_api_key="fake", redis_startup_check=True)
    monkeypatch.setattr("ocr_service.routers.system.get_settings", lambda: settings)

    async def fake_verify(_client, _timeout=1.0):
        return {"ok": False, "error": "no"}

    monkeypatch.setattr(
        "ocr_service.utils.redis_factory.verify_redis_connection",
        fake_verify,
    )

    res = await health_check()
    assert res.status == "degraded"
    assert res.components["redis"]["ok"] is False


@pytest.mark.asyncio
async def test_health_redis_check_skipped(monkeypatch):
    settings = Settings(ocr_api_key="fake", redis_startup_check=False)
    monkeypatch.setattr("ocr_service.routers.system.get_settings", lambda: settings)

    async def fake_verify(_client, _timeout=1.0):
        raise RuntimeError("verify should be skipped")

    monkeypatch.setattr(
        "ocr_service.utils.redis_factory.verify_redis_connection",
        fake_verify,
    )

    res = await health_check()
    assert res.status == "ok"
    assert res.components["redis"]["ok"] is True
    assert res.components["redis"]["skipped"] is True
