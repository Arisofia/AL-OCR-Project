"""
Advanced pixel reconstruction module using Vision LLMs.

This module provides high-depth layer elimination and content reconstruction
for obscured documents using multiple AI providers.
"""

import logging
from typing import Any, Optional

from ocr_service.config import get_settings

from .ai_providers import (
    AIProviderError,
    GeminiVisionProvider,
    HuggingFaceVisionProvider,
    OpenAIVisionProvider,
    VisionProvider,
)

logger = logging.getLogger("ocr-service.advanced-recon")


class AdvancedPixelReconstructor:
    """
    Leverages Vision LLMs to perform deep pixel-by-pixel reconstruction
    and layer elimination for obscured documents.
    """

    def __init__(self, providers: Optional[dict[str, VisionProvider]] = None):
        """
        Initializes the reconstructor with available AI providers.
        """
        self.settings = get_settings()
        if providers:
            self.providers = providers
        else:
            # Default initialization based on settings
            self.providers = {}
            if self.settings.openai_api_key:
                self.providers["openai"] = OpenAIVisionProvider(
                    self.settings.openai_api_key
                )
            if self.settings.gemini_api_key:
                self.providers["gemini"] = GeminiVisionProvider(
                    self.settings.gemini_api_key
                )
            if self.settings.hugging_face_hub_token:
                self.providers["huggingface"] = HuggingFaceVisionProvider(
                    self.settings.hugging_face_hub_token
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

        Args:
            image_bytes (bytes): The image data to reconstruct.
            provider (str): The AI provider to use (default: "openai").
            context (Optional[dict[str, Any]]): Optional context dictionary. Expected keys:
                - "font_metadata": str, metadata about document fonts
                - "accuracy_score": str or float, previous reconstruction accuracy
            fallback (bool): Whether to fallback to other providers if the primary fails.

        Returns:
            dict[str, Any]: Reconstruction result or error details.
        """
        primary = provider
        if primary not in self.providers:
            # Try to find any available provider if the requested one is missing
            if not self.providers:
                return {"error": "No AI providers configured"}
            primary = next(iter(self.providers.keys()))
            logger.info(
                "Requested provider %s unavailable, using %s", provider, primary
            )

        prompt = (
            "Analyze this document image. Identify any obscured, pixelated, "
            "or layered parts. Reconstruct the underlying text and structure "
            "pixel-by-pixel in your understanding and provide the full "
            "corrected text. Eliminate any noise or overlays."
        )

        if context:
            font_meta = context.get("font_metadata", None)
            acc_score = context.get("accuracy_score", None)
            if font_meta is None:
                logger.warning("Context missing 'font_metadata' key; using default.")
                font_meta = "No font metadata available"
            if acc_score is None:
                logger.warning("Context missing 'accuracy_score' key; using default.")
                acc_score = "N/A"
            prompt += (
                f"\n\nContext from similar documents: {font_meta}. "
                f"Accuracy of previous similar reconstructions: {acc_score}."
            )

        try:
            return await self.providers[primary].reconstruct(image_bytes, prompt)
        except AIProviderError as e:
            logger.warning("Primary provider %s failed: %s", primary, e)
            if fallback:
                return await self._try_fallback(image_bytes, prompt, exclude=primary)
            return {"error": str(e), "details": e.details}
        except RuntimeError as e:
            logger.error("Unexpected failure in %s: %s", primary, e)
            if fallback:
                return await self._try_fallback(image_bytes, prompt, exclude=primary)
            return {"error": f"Internal error: {e}"}

    async def _try_fallback(
        self, image_bytes: bytes, prompt: str, exclude: str
