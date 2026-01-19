"""
Advanced pixel reconstruction module using Vision LLMs.

This module provides high-depth layer elimination and content reconstruction
for obscured documents using multiple AI providers.
"""

# pylint: disable=broad-except,too-few-public-methods

import logging
from typing import Any, Optional

import httpx
from ocr_service.config import get_settings

__all__ = ["AdvancedPixelReconstructor"]

from .ai_providers import (
    AIProviderError,
    BaseVisionProvider,
    GeminiVisionProvider,
    HuggingFaceVisionProvider,
    OpenAIVisionProvider,
    VisionProvider,
)

logger = logging.getLogger("ocr-service.advanced-recon")

BASE_RECON_PROMPT = (
    "Analyze this document image. Identify any obscured, pixelated, "
    "or layered parts. Reconstruct the underlying text and structure "
    "pixel-by-pixel in your understanding and provide the full "
    "corrected text. Eliminate any noise or overlays."
)


class AdvancedPixelReconstructor:
    """
    Leverages Vision LLMs to perform deep pixel-by-pixel reconstruction
    and layer elimination for obscured documents.
    """

    def __init__(
        self,
        providers: Optional[dict[str, VisionProvider]] = None,
        client: Optional[httpx.AsyncClient] = None,
    ):
        """
        Initializes the reconstructor with available AI providers.
        """
        self.settings = get_settings()
        self._client = client
        if providers:
            self.providers = providers
        else:
            self._initialize_default_providers()

    async def close(self) -> None:
        """Closes all underlying providers."""
        for provider in self.providers.values():
            if isinstance(provider, BaseVisionProvider):
                await provider.close()

    def _initialize_default_providers(self) -> None:
        """Initializes providers based on environment settings."""
        self.providers = {}
        if self.settings.openai_api_key:
            self.providers["openai"] = OpenAIVisionProvider(
                self.settings.openai_api_key, client=self._client
            )
        if self.settings.gemini_api_key:
            self.providers["gemini"] = GeminiVisionProvider(
                self.settings.gemini_api_key, client=self._client
            )
        if self.settings.hugging_face_hub_token:
            self.providers["huggingface"] = HuggingFaceVisionProvider(
                self.settings.hugging_face_hub_token, client=self._client
            )

    async def reconstruct_with_ai(
        self,
        image_bytes: bytes,
        provider: str = "openai",
        context: Optional[dict[str, Any]] = None,
        fallback: bool = True,
    ) -> dict[str, Any]:
        """
        Uses advanced AI models to reconstruct obscured content.
        """
        primary = self._get_primary_provider(provider)
        if not primary:
            return {"error": "No AI providers configured"}

        prompt = self._build_prompt(context)

        try:
            return await self.providers[primary].reconstruct(
                image_bytes, prompt
            )
        except (AIProviderError, httpx.HTTPError, Exception) as e:
            logger.warning(
                "Primary provider %s failed (%s): %s",
                primary,
                type(e).__name__,
                e,
            )
            if fallback:
                return await self._try_fallback(
                    image_bytes, prompt, exclude=primary
                )

            return self._format_error(e)

    def _get_primary_provider(self, requested: str) -> Optional[str]:
        """Resolves the primary provider, falling back to any available if missing."""
        if requested in self.providers:
            return requested

        if not self.providers:
            return None

        first_available = next(iter(self.providers.keys()))
        logger.info(
            "Requested provider %s unavailable, using %s",
            requested,
            first_available,
        )
        return first_available

    def _build_prompt(self, context: Optional[dict[str, Any]]) -> str:
        """Constructs the reconstruction prompt with optional context."""
        prompt = BASE_RECON_PROMPT
        if context:
            font_meta = context.get(
                "font_metadata", "No font metadata available"
            )
            acc_score = context.get("accuracy_score", "N/A")
            prompt += (
                f"\n\nContext from similar documents: {font_meta}. "
                f"Accuracy of previous similar reconstructions: {acc_score}."
            )
        return prompt

    def _format_error(self, e: Exception) -> dict[str, Any]:
        """Formats exceptions into a standardized error response."""
        if isinstance(e, AIProviderError):
            return {"error": str(e), "details": e.details}
        if isinstance(e, httpx.HTTPError):
            return {"error": f"Network error: {e}"}
        return {"error": f"Internal error: {e}"}

    async def _try_fallback(
        self, image_bytes: bytes, prompt: str, exclude: str
    ) -> dict[str, Any]:
        """
        Attempts to use an alternative provider when the primary fails.
        """
        for name, provider in self.providers.items():
            if name == exclude:
                continue

            logger.info("Attempting fallback to %s", name)
            try:
                return await provider.reconstruct(image_bytes, prompt)
            except AIProviderError as e:
                logger.warning("Fallback to %s failed: %s", name, e)
            except Exception as e:  # noqa: BLE001
                logger.error("Unexpected fallback failure in %s: %s", name, e)

        return {"error": "All AI providers failed"}
