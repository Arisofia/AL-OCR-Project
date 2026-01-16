from fastapi.testclient import TestClient
from ocr_service.main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] in ("healthy", "degraded")
    assert "timestamp" in data
    assert "services" in data
    assert "s3" in data["services"]
