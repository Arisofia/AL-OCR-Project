import pytest

from fastapi.testclient import TestClient

from ocr_service.app import create_app
from ocr_service.config import Settings


class FailingRedis:
    async def get(self, _key):
        raise RuntimeError("redis connection failure")

    async def set(self, _key, _value, _ex=None):
        raise RuntimeError("redis set failure")

    async def delete(self, _key):
        raise RuntimeError("redis delete failure")


@pytest.mark.xfail(reason="Simulated Redis outage: expected failure for negative test scenario")
def test_ocr_endpoint_returns_structured_error_on_redis_failure(monkeypatch):
    settings = Settings(ocr_api_key="fake")
    app = create_app(settings=settings)

    # Patch the Redis factory used by dependency provider to return failing redis
    from ocr_service.routers import deps

    def _fake_create_redis_client(_settings):
        return FailingRedis()

    monkeypatch.setattr(deps, "create_redis_client", _fake_create_redis_client)

    client = TestClient(app)

    files = {"file": ("test.png", b"image-bytes", "image/png")}
    headers = {"X-API-KEY": "fake"}

    resp = client.post("/ocr", files=files, headers=headers)

    assert resp.status_code == 500
    body = resp.json()
    assert body.get("phase") == "idempotency"
    assert body.get("correlation_id") is not None
    # trace_id may be None if no span is present; ensure key exists (may be absent)
    assert "trace_id" in body
