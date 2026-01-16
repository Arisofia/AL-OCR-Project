import os
import sys


import requests


def test_health_check():
    # Default to local FastAPI dev server
    url = os.environ.get("OCR_HEALTH_URL", "http://127.0.0.1:8000/health")
    print(f"Testing health endpoint: {url}")
    resp = requests.get(url, timeout=10)

    assert (
        resp.status_code == 200
    ), f"Health endpoint failed: {resp.status_code} {resp.text}"

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
    print("Health check passed:", data)


if __name__ == "__main__":
    try:
        test_health_check()
    except AssertionError as e:
        print(f"Health check failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)
