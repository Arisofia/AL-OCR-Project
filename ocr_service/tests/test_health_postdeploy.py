import os

import pytest
import requests  # type: ignore
from requests.exceptions import RequestException


@pytest.mark.skipif(
    os.environ.get("SKIP_HEALTH_CHECK") == "true", reason="Skipping post-deploy test"
)
def test_health_check():
    # Default to local FastAPI dev server
    url = os.environ.get("OCR_HEALTH_URL", "http://127.0.0.1:8000/health")
    try:
        resp = requests.get(url, timeout=5)
    except RequestException:
        pytest.skip(f"Health check server not reachable at {url}")

    # Ensure a successful status
    resp.raise_for_status()

    data = resp.json()
    # Check for required keys
    assert "status" in data, f"Missing 'status' in health response: {data}"
    assert data["status"] in (
        "healthy",
        "degraded",
    ), f"Unexpected status: {data['status']}"
    assert "services" in data, f"Missing 'services' in health response: {data}"
    expected_services = {"s3", "supabase", "openai", "gemini"}
    assert isinstance(
        data["services"], dict
    ), f"'services' should be a dict, got {type(data['services'])}"
    actual_services = set(data["services"].keys())
    missing = expected_services - actual_services
    assert not missing, f"Missing services: {missing}. Found: {actual_services}"
