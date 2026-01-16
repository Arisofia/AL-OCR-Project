"""
Provides abstractions and implementations for various AI Vision providers.
Used for advanced document reconstruction and verification.
"""

import base64
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict

import httpx

logger = logging.getLogger("ocr-service.ai-providers")


class VisionProvider(ABC):
    """
    Abstract base class for vision-based AI providers.
    """

    @abstractmethod
    async def reconstruct(self, image_bytes: bytes, prompt: str) -> Dict[str, Any]:
        """
        Processes an image with a prompt to reconstruct or analyze content.
        """


class OpenAIVisionProvider(VisionProvider):
    """
    Implementation of VisionProvider using OpenAI's GPT-4o model.
    """

    def __init__(self, api_key: str):
        self.api_key = api_key

    async def reconstruct(self, image_bytes: bytes, prompt: str) -> Dict[str, Any]:
        """
        Sends an image to OpenAI for reconstruction.
        """
        base64_image = base64.b64encode(image_bytes).decode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
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
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=60.0,
                )
                response.raise_for_status()
                data = response.json()
                return {
                    "text": data["choices"][0]["message"]["content"],
                    "model": "gpt-4o",
                }
            except httpx.HTTPError as e:
                logger.error("OpenAI Vision HTTP error: %s", e)
                return {"error": f"OpenAI HTTP error: {e}"}
            except (KeyError, IndexError) as e:
                logger.error("OpenAI Vision response parsing failed: %s", e)
                return {"error": "Unexpected response format from OpenAI"}
            except Exception as e:
                logger.error("OpenAI Vision unexpected error: %s", e)
                return {"error": str(e)}


class GeminiVisionProvider(VisionProvider):
    """
    Implementation of VisionProvider using Google's Gemini Flash model.
    """

    def __init__(self, api_key: str):
        self.api_key = api_key

    async def reconstruct(self, image_bytes: bytes, prompt: str) -> Dict[str, Any]:
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
        except ImportError:
            logger.error("google-generativeai package not installed")
            return {"error": "Gemini provider not available: package missing"}
        except AttributeError as e:
            logger.error("Gemini Vision response parsing failed: %s", e)
            return {"error": "Invalid response structure from Gemini"}
        except Exception as e:
            logger.error("Gemini Vision unexpected error: %s", e)
            return {"error": str(e)}


class HuggingFaceVisionProvider(VisionProvider):
    """
    Implementation of VisionProvider using Hugging Face Inference Router.

    Uses the new router host (https://router.huggingface.co) and retries on
    rate-limits (429) with exponential backoff. Falls back to a simple JSON
    response extraction.
    """

    def __init__(self, token: str, model: str = "runwayml/stable-diffusion-v1-5"):
        self.token = token
        self.model = model

    async def reconstruct(self, image_bytes: bytes, prompt: str) -> Dict[str, Any]:
        import asyncio

        base64_image = base64.b64encode(image_bytes).decode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.token}",
        }
        url = f"https://router.huggingface.co/models/{self.model}"
        payload = {
            "inputs": {
                "image": f"data:image/jpeg;base64,{base64_image}",
                "prompt": prompt,
            }
        }

        max_attempts = 3
        attempt = 0
        while attempt < max_attempts:
            async with httpx.AsyncClient() as client:
                try:
                    response = await client.post(
                        url, headers=headers, json=payload, timeout=60.0
                    )
                except httpx.HTTPError as e:
                    logger.error(
                        "HuggingFace HTTP error on attempt %s: %s", attempt + 1, e
                    )
                    return {"error": f"HuggingFace HTTP error: {e}"}

            # Retry on 429 with exponential backoff
            if response.status_code == 429:
                backoff = 2**attempt
                logger.warning(
                    "HuggingFace rate limited, retrying in %s seconds (attempt %s)",
                    backoff,
                    attempt + 1,
                )
                await asyncio.sleep(backoff)
                attempt += 1
                continue

            try:
                response.raise_for_status()
                data = response.json()
                # Many HF vision/text models return different shapes; be permissive
                text = None
                if isinstance(data, dict):
                    text = (
                        data.get("generated_text")
                        or data.get("text")
                        or data.get("result")
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
                    text = data
                return {"text": text, "model": self.model}
            except httpx.HTTPError as e:
                logger.error("HuggingFace response parsing failed: %s", e)
                return {"error": f"HuggingFace HTTP error: {e}"}
            except Exception as e:
                logger.error("HuggingFace unexpected error: %s", e)
                return {"error": str(e)}

        # If we've exhausted attempts without success, return an error
        return {"error": "Exceeded maximum retry attempts due to rate limiting"}