from fastapi.testclient import TestClient
from main import app


def test_recon_status_endpoint():
    client = TestClient(app)
    resp = client.get("/recon/status")
    assert resp.status_code == 200
    data = resp.json()
    assert "reconstruction_enabled" in data
    assert "package_installed" in data
    assert "package_version" in data
