"""Health endpoint tests for app factory output."""

from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from ocr_service.app import create_app
from ocr_service.config import Settings
import ocr_service.routers.system as system_router


def test_health_check_ok(monkeypatch):
    settings = Settings(ocr_api_key="fake")
    app = create_app(settings=settings)

    # Patch redis and storage checks to return healthy
    def _fake_get_redis_client(_settings):
        return MagicMock()

    def _fake_verify_redis_connection(_client):
        return {"ok": True, "latency_ms": 1}

    monkeypatch.setattr(system_router, "get_redis_client", _fake_get_redis_client)
    monkeypatch.setattr(
        system_router, "verify_redis_connection", _fake_verify_redis_connection
    )

    class DummyStorage:
        """Storage stub that always reports healthy connectivity."""

        def check_connection(self):
            return True

    monkeypatch.setattr(
        "ocr_service.routers.system.StorageService",
        DummyStorage,
    )

    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["components"]["redis"]["ok"] is True
    assert body["components"]["s3"]["ok"] is True
