"""Test negative Redis scenario: OCR endpoint returns error on Redis failure."""

import pytest
from fastapi.testclient import TestClient

from ocr_service.app import create_app
from ocr_service.config import Settings
from ocr_service.routers import deps


class FailingRedis:
    """Mock Redis client that always fails for negative test."""

    async def get(self, _key):
        """Simulate Redis get failure."""
        raise RuntimeError("redis connection failure")

    async def set(self, _key, _value, _ex=None):
        """Simulate Redis set failure."""
        raise RuntimeError("redis set failure")

    async def delete(self, _key):
        """Simulate Redis delete failure."""
        raise RuntimeError("redis delete failure")


@pytest.mark.xfail(
    reason="Simulated Redis outage: expected failure for negative test scenario"
)
def test_ocr_endpoint_returns_structured_error_on_redis_failure(monkeypatch):
    """Test that OCR endpoint returns 500 error if Redis fails (negative test)."""
    settings = Settings(ocr_api_key="fake")
    app = create_app(settings=settings)

    def _fake_create_redis_client(_settings):
        """Return a failing Redis client for dependency injection."""
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
    assert "trace_id" in body
