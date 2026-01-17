import asyncio
import base64
import logging
from abc import ABC, abstractmethod
from typing import Any, Optional, Union, cast

import httpx

logger = logging.getLogger("ocr-service.ai-providers")


class AIProviderError(Exception):
    """Base class for AI provider errors."""

    def __init__(self, message: str, details: Optional[dict[str, Any]] = None):
        super().__init__(message)
        self.details = details or {}


class ProviderConfigError(AIProviderError):
    """Raised when a provider is misconfigured (e.g., missing API key)."""


class ProviderRuntimeError(AIProviderError):
    """Raised during API execution failures (e.g., HTTP errors, parsing)."""


class VisionProvider(ABC):
    """
    Abstract base class for vision-based AI providers.
    """

    @abstractmethod
    async def reconstruct(self, image_bytes: bytes, prompt: str) -> dict[str, Any]:
        """
        Processes an image with a prompt to reconstruct or analyze content.
        Raises AIProviderError on failure.
        """


class BaseVisionProvider(VisionProvider, ABC):
    """
    Base class providing common utilities for VisionProviders, such as retries.
    """

    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries

    async def _request_with_retry(
        self,
        url: str,
        headers: dict[str, str],
        json_payload: dict[str, Any],
        method: str = "POST",
        timeout: float = 60.0,
    ) -> Union[dict[str, Any], list[Any]]:
        """
        Internal helper to perform HTTP requests with exponential backoff on 429s.
        """
        attempt = 0
        while attempt < self.max_retries:
            async with httpx.AsyncClient() as client:
                try:
                    response = await client.request(
                        method,
                        url,
                        headers=headers,
                        json=json_payload,
                        timeout=timeout,
                    )
                except httpx.HTTPError as e:
                    logger.error("HTTP error on attempt %s: %s", attempt + 1, e)
                    if attempt == self.max_retries - 1:
                        raise ProviderRuntimeError(
                            f"HTTP error after {self.max_retries} attempts: {e}"
                        ) from e
                    attempt += 1
                    await asyncio.sleep(2**attempt)
                    continue

            if response.status_code == 429:
                backoff = 2**attempt
                logger.warning("Rate limited, retrying in %s seconds", backoff)
                await asyncio.sleep(backoff)
                attempt += 1
                continue

            try:
                response.raise_for_status()
                return cast(Union[dict[str, Any], list[Any]], response.json())
            except httpx.HTTPStatusError as e:
                response_body = self._get_response_body(e)
                logger.error("HTTP status error: %s | body: %s", e, response_body)
                raise ProviderRuntimeError(
                    f"HTTP status error: {e.response.status_code}",
                    details={
                        "status_code": response.status_code,
                        "body": response_body,
                    },
                ) from e

        raise ProviderRuntimeError("Exceeded maximum retry attempts")

    def _get_response_body(self, exc: httpx.HTTPStatusError) -> Optional[Any]:
        """Extracts JSON or text from an HTTPStatusError response."""
        resp = getattr(exc, "response", None)
        if resp is not None:
            try:
                return resp.json()
            except Exception:
                try:
                    return resp.text
                except Exception:
                    return None
        return None


class OpenAIVisionProvider(BaseVisionProvider):
    """
    Implementation of VisionProvider using OpenAI's GPT-4o model.
    """

    def __init__(self, api_key: str, max_retries: int = 3):
        super().__init__(max_retries=max_retries)
        self.api_key = api_key

    async def reconstruct(self, image_bytes: bytes, prompt: str) -> dict[str, Any]:
        """
        Sends an image to OpenAI for reconstruction.
        """
        base64_image = base64.b64encode(image_bytes).decode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        request_payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
            "max_tokens": 2000,
        }

        data = await self._request_with_retry(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json_payload=request_payload,
        )

        if not isinstance(data, dict):
            raise ProviderRuntimeError("Unexpected response format from OpenAI")

        try:
            return {
                "text": data["choices"][0]["message"]["content"],
                "model": "gpt-4o",
            }
        except (KeyError, IndexError) as e:
            logger.error("OpenAI response parsing failed: %s", e)
            raise ProviderRuntimeError("Failed to parse OpenAI response") from e


class GeminiVisionProvider(VisionProvider):
    """
    Implementation of VisionProvider using Google's Gemini Flash model.
    """

    def __init__(self, api_key: str):
        self.api_key = api_key

    async def reconstruct(self, image_bytes: bytes, prompt: str) -> dict[str, Any]:
        """
        Sends an image to Gemini for reconstruction.
        """
        try:
            import google.generativeai as genai

            genai.configure(api_key=self.api_key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            image_part = {"mime_type": "image/jpeg", "data": image_bytes}
            response = await model.generate_content_async([prompt, image_part])
            return {"text": response.text, "model": "gemini-1.5-flash"}
        except ImportError as e:
            logger.error("google-generativeai package not installed")
            raise ProviderConfigError("Gemini package missing") from e
        except AttributeError as e:
            logger.error("Gemini Vision response parsing failed: %s", e)
            raise ProviderRuntimeError("Invalid response structure from Gemini") from e
        except Exception as e:
            logger.error("Gemini Vision unexpected error: %s", e)
            raise ProviderRuntimeError(str(e)) from e


class HuggingFaceVisionProvider(BaseVisionProvider):
    """
    Implementation of VisionProvider using Hugging Face Inference Router.
    """

    def __init__(
        self,
        token: str,
        model: str = "runwayml/stable-diffusion-v1-5",
        max_retries: int = 3,
    ):
        super().__init__(max_retries=max_retries)
        self.token = token
        self.model = model

    async def reconstruct(self, image_bytes: bytes, prompt: str) -> dict[str, Any]:
        base64_image = base64.b64encode(image_bytes).decode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.token}",
        }
        url = f"https://router.huggingface.co/models/{self.model}"
        request_payload = {
            "inputs": {
                "image": f"data:image/jpeg;base64,{base64_image}",
                "prompt": prompt,
            }
        }

        data = await self._request_with_retry(
            url, headers=headers, json_payload=request_payload
        )

        try:
            # Many HF vision/text models return different shapes; be permissive
            text = None
            if isinstance(data, dict):
                text = (
                    data.get("generated_text") or data.get("text") or data.get("result")
                )
            elif isinstance(data, list) and data:
                first = data[0]
                if isinstance(first, dict):
                    text = (
                        first.get("generated_text")
                        or first.get("text")
                        or first.get("result")
                    )
                elif isinstance(first, str):
                    text = first

            if text is None:
                text = str(data)

            return {"text": text, "model": self.model}
        except Exception as e:
            logger.error("HuggingFace response parsing failed: %s", e)
            raise ProviderRuntimeError(f"Unexpected response format: {e}") from e
