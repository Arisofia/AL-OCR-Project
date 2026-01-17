from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from ocr_service.config import get_settings
from ocr_service.main import app, limiter


def _exhaust_rate_limit(client, url, body, headers, max_attempts=50):
    """Helper to exhaust rate limit and return the 429 response."""
    for _ in range(max_attempts):
        resp = client.post(url, json=body, headers=headers)
        if resp.status_code == 429:
            return resp
    return None


def _verify_log_emission(mock_logger, expected_path):
    """Helper to verify the structured log was emitted."""
    log_calls = mock_logger.warning.call_args_list
    was_logged = any(
        args[0] == "Rate limit exceeded"
        and kwargs.get("extra", {}).get("path") == expected_path
        for args, kwargs in log_calls
    )
    assert was_logged, "Expected structured rate limit warning log was not emitted"


def test_rate_limit_smoke_triggers_handler():
    """Make repeated requests to a rate-limited endpoint and ensure our
    handler runs on 429.
    """
    settings = get_settings()
    settings.s3_bucket_name = "test-bucket"

    # Reset limiter storage to ensure a clean state for this test
    getattr(limiter, "reset", lambda: None)()

    client = TestClient(app)
    headers = {str(settings.api_key_header_name): str(settings.ocr_api_key)}
    body = {
        "key": "uploads/test.png",
        "content_type": "image/png",
    }

    # Patch S3 client so presign succeeds during the first calls
    with patch("boto3.client") as mock_boto, patch(
        "ocr_service.utils.limiter.logger"
    ) as mock_logger:
        mock_s3 = MagicMock()
        mock_s3.generate_presigned_post.return_value = {
            "url": "https://s3.example.com/bucket",
            "fields": {"key": "uploads/test.png", "policy": "abc"},
        }
        mock_boto.return_value = mock_s3

        rate_limited_response = _exhaust_rate_limit(client, "/presign", body, headers)

        assert (
            rate_limited_response is not None
        ), "Expected at least one 429 response from rate limiter"
        assert "detail" in rate_limited_response.json()
        # Validate the handler-specific payload (adjust if message changes)
        assert rate_limited_response.json().get("detail") == "Rate limit exceeded"

        # Assert our enhanced handler logged the rate limiting occurrence
        _verify_log_emission(mock_logger, "/presign")
