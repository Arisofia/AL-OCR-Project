"""
Advanced pixel reconstruction module using Vision LLMs.

This module provides high-depth layer elimination and content reconstruction
for obscured documents using multiple AI providers.
"""

import logging
from typing import Optional, Dict, Any

from config import get_settings
from .ai_providers import VisionProvider, OpenAIVisionProvider, GeminiVisionProvider

logger = logging.getLogger("ocr-service.advanced-recon")


class AdvancedPixelReconstructor:
    """
    Leverages Vision LLMs to perform deep pixel-by-pixel reconstruction
    and layer elimination for obscured documents.
    """

    def __init__(self, providers: Optional[Dict[str, VisionProvider]] = None):
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

    async def reconstruct_with_ai(
        self,
        image_bytes: bytes,
        provider: str = "openai",
        context: Optional[Dict[str, Any]] = None,
        fallback: bool = True
    ) -> Dict[str, Any]:
        """
        Uses advanced AI models to 'see' through layers and reconstruct content.
        """
        primary = provider
        if primary not in self.providers:
            # Try to find any available provider if the requested one is missing
            if not self.providers:
                return {"error": "No AI providers configured"}
            primary = list(self.providers.keys())[0]
            logger.info(
                "Requested provider %s unavailable, using %s",
                provider,
                primary
            )

        prompt = (
            "Analyze this document image. Identify any obscured, pixelated, "
            "or layered parts. Reconstruct the underlying text and structure "
            "pixel-by-pixel in your understanding and provide the full corrected text. "
            "Eliminate any noise or overlays."
        )

        if context:
            font_meta = context.get('font_metadata', 'No font metadata available')
            acc_score = context.get('accuracy_score', 'N/A')
            prompt += (
                f"\n\nContext from similar documents: {font_meta}. "
                f"Accuracy of previous similar reconstructions: {acc_score}."
            )

        try:
            result = await self.providers[primary].reconstruct(image_bytes, prompt)
            if "error" in result and fallback:
                return await self._try_fallback(image_bytes, prompt, exclude=primary)
            return result
        except Exception as e:
            logger.error("Primary provider %s failed: %s", primary, e)
            if fallback:
                return await self._try_fallback(image_bytes, prompt, exclude=primary)
            return {"error": str(e)}

    async def _try_fallback(
        self,
        image_bytes: bytes,
        prompt: str,
        exclude: str
    ) -> Dict[str, Any]:
        """
        Attempts to use an alternative provider when the primary fails.
        """
        for name, provider in self.providers.items():
            if name != exclude:
                logger.info("Attempting fallback to %s", name)
                try:
                    result = await provider.reconstruct(image_bytes, prompt)
                    if "error" not in result:
                        return result
                except Exception as e:
                    logger.warning("Fallback to %s failed: %s", name, e)

        return {"error": "All AI providers failed"}
