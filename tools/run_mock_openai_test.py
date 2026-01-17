import asyncio
import logging
from pathlib import Path

import httpx

from ocr_service.modules.ai_providers import OpenAIVisionProvider

# Create a tiny dummy image (PNG) using builtins to avoid extra deps
img_path = Path("test_image.png")
if not img_path.exists():
    from PIL import Image

    Image.new("RGB", (10, 10), color=(255, 255, 255)).save(img_path)

# Setup logging to capture provider logs
logger = logging.getLogger("ocr-service.ai-providers")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logger.addHandler(handler)


# Mock httpx.AsyncClient to simulate a 403 response
class _MockResponse:
    def __init__(self):
        self.status_code = 403

    def raise_for_status(self):
        raise httpx.HTTPStatusError("403 Forbidden", request=None, response=self)

    def json(self):
        return {"error": "quota_exceeded", "message": "You have exceeded your quota."}


class _MockClient:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, url, headers=None, json=None, timeout=None):
        # Log the URL for clarity and return a mock response object
        logger.debug("MockClient.post called for URL: %s", url)
        return _MockResponse()


async def main():
    # Patch httpx.AsyncClient to our mock class (direct replacement)
    original = httpx.AsyncClient
    httpx.AsyncClient = _MockClient

    try:
        provider = OpenAIVisionProvider(api_key="fake-api-key")
        with open(img_path, "rb") as f:
            image_bytes = f.read()

        res = await provider.reconstruct(image_bytes, "Test prompt")
        print("Provider result:", res)
    finally:
        httpx.AsyncClient = original


if __name__ == "__main__":
    asyncio.run(main())
