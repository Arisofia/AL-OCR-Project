import asyncio

import httpx

from ocr_service.modules.ai_providers import HuggingFaceVisionProvider


class _MockResponse:
    def __init__(self, status_code: int, payload):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                "status",
                request=httpx.Request("POST", "https://example.com"),
                response=httpx.Response(self.status_code),
            )


class _MockClient:
    def __init__(self, responses):
        self._responses = responses
        self.calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, url, headers=None, json=None, timeout=None):
        # Keep the timeout parameter to match httpx.AsyncClient.post signature
        _ = timeout
        self.calls.append({"url": url, "headers": headers, "json": json})
        return self._responses.pop(0)


def test_huggingface_provider_uses_router_and_auth_and_retries(monkeypatch):
    # First response is 429, second is 200
    responses = [_MockResponse(429, {}), _MockResponse(200, {"generated_text": "ok"})]
    mock_client = _MockClient(responses)

    # Patch httpx.AsyncClient to use our mock client
    monkeypatch.setattr(httpx, "AsyncClient", lambda: mock_client)

    provider = HuggingFaceVisionProvider(token="fake-token", model="test-model")

    res = asyncio.run(provider.reconstruct(b"bytes", "a prompt"))

    assert res["text"] == "ok"
    assert res["model"] == "test-model"

    # Ensure the first call used the router host and included Authorization header
    first_call = mock_client.calls[0]
    assert first_call["url"].startswith(
        "https://router.huggingface.co/models/test-model"
    )
    assert first_call["headers"]["Authorization"] == "Bearer fake-token"


def test_huggingface_provider_handles_nonstandard_payload(monkeypatch):
    responses = [_MockResponse(200, ["result-text"])]
    mock_client = _MockClient(responses)
    monkeypatch.setattr(httpx, "AsyncClient", lambda: mock_client)

    provider = HuggingFaceVisionProvider(token="fake-token", model="test-model")
    res = asyncio.run(provider.reconstruct(b"bytes", "a prompt"))

    assert res["text"] == "result-text"
