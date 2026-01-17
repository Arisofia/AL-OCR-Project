import asyncio
import logging
from pathlib import Path

import httpx

from ocr_service.modules.ai_providers import OpenAIVisionProvider

# Use a stable, minimal binary file for image bytes (no external deps)
img_path = Path("test_image.bin")
if not img_path.exists():
    img_path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)

logger = logging.getLogger("ocr-service.ai-providers")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logger.handlers.clear()
logger.addHandler(handler)


class _MockResponse:
    def __init__(
        self,
        status_code: int = 403,
        body: dict | None = None,
        method: str = "POST",
        url: str = "https://api.openai.com/v1/chat/completions",
    ):
        """
        Initialize a mock response object.

        Args:
            status_code (int, optional): The HTTP status code of the response.
                Defaults to 403.
            body (dict | None, optional): The response body. Defaults to None.
            method (str, optional): HTTP method for error context. Defaults to POST.
            url (str, optional): URL for error context.
                Defaults to OpenAI completions endpoint.

        If body is None, it will be set to a default quota exceeded response.
        """
        self.status_code = status_code
        self._body = body or {
            "error": "quota_exceeded",
            "message": "You have exceeded your quota.",
        }
        self._method = method
        self._url = url

    def raise_for_status(self):
        """Raise an HTTPStatusError if the mock response indicates an error.

        This emulates httpx.Response.raise_for_status by turning error-like
        status codes into exceptions for test scenarios.

        Raises:
            httpx.HTTPStatusError: If the status code is 400 or higher.
        """
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                f"{self.status_code} error",
                request=httpx.Request(self._method, self._url),
                response=httpx.Response(self.status_code),
            )

    def json(self):
        return self._body


class _MockAsyncClient:
    def __init__(self, status_code: int = 403, body: dict | None = None):
        self._status_code = status_code
        self._body = body

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, url, headers=None, json=None, timeout=None):
        logger.debug("MockAsyncClient.post called for URL: %s", url)
        logger.debug("Request headers: %s", headers)
        logger.debug("Request payload keys: %s", list((json or {}).keys()))
        # Use the timeout parameter if provided to simulate delay
        if timeout:
            await asyncio.sleep(timeout)
        return _MockResponse(status_code=self._status_code, body=self._body)


async def run_scenario(name: str, status_code: int, body: dict | None = None):
    original_client = httpx.AsyncClient
    httpx.AsyncClient = lambda *a, **k: _MockAsyncClient(
        status_code=status_code, body=body
    )

    try:
        logger.info("=== Running scenario: %s (HTTP %s) ===", name, status_code)
        provider = OpenAIVisionProvider(api_key="fake-api-key")
        image_bytes = img_path.read_bytes()
        result = await provider.reconstruct(image_bytes, "Test prompt")
        print(f"[{name}] Provider result:", result)
    except Exception as exc:
        logger.exception("[%s] Provider raised exception", name)
        print(f"[{name}] Exception:", repr(exc))
    finally:
        httpx.AsyncClient = original_client


async def main():
    await run_scenario(
        name="QuotaExceeded403",
        status_code=403,
        body={
            "error": {
                "type": "insufficient_quota",
                "message": "You have exceeded your quota.",
            }
        },
    )
    await run_scenario(
        name="ServerError500",
        status_code=500,
        body={"error": {"type": "server_error", "message": "Internal error"}},
    )


if __name__ == "__main__":
    asyncio.run(main())
