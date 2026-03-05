"""
Card number validation helpers.

This module validates card numbers using a local Luhn check and (optionally)
enriches results with external BIN metadata.

Security note: This module never completes or generates PAN digits. It only
validates numbers provided by upstream OCR and should be used with tokenization
flows (Stripe/Adyen/etc.) rather than storing PAN.
"""

import asyncio
import logging
import os
from typing import Any, Optional

import httpx

from ocr_service.modules.document_intelligence import DocumentIntelligence

logger = logging.getLogger("ocr-service.card-validator")


class CardValidator:
    """Validate card numbers with local checks and optional external lookups."""

    def __init__(self, provider: Optional[str] = None, api_key: Optional[str] = None):
        resolved_provider = (
            provider or os.getenv("CARD_VALIDATOR_PROVIDER") or "binlist"
        ).strip()
        self.provider = resolved_provider.lower()
        self.api_key = api_key or os.getenv("CARD_VALIDATOR_API_KEY")
        self.client = httpx.AsyncClient(timeout=5.0)

    @staticmethod
    def _compact_digits(value: str) -> str:
        return "".join(ch for ch in (value or "") if ch.isdigit())

    @staticmethod
    def _local_luhn_and_brand(clean_number: str) -> dict[str, Any]:
        if not clean_number.isdigit() or not 13 <= len(clean_number) <= 19:
            return {"valid": False, "error": "Invalid length"}
        if not DocumentIntelligence.is_valid_luhn(clean_number):
            return {"valid": False, "error": "Local Luhn check failed"}
        return {
            "valid": True,
            "brand": DocumentIntelligence.guess_card_brand(clean_number),
        }

    async def validate(self, card_number: str) -> dict[str, Any]:
        """
        Validate a card number.

        Performs local Luhn validation first. If a provider is configured, it may
        also perform external BIN enrichment/verification.
        """
        clean_number = self._compact_digits(card_number)
        local = self._local_luhn_and_brand(clean_number)
        if not local.get("valid"):
            return local

        brand = local.get("brand", "unknown")

        _provider_handlers = {
            "mock": self._validate_mock,
            "numbrify": self._validate_numbrify,
            "binlist": self._validate_binlist,
            "cardio": self._validate_cardio,
            "worldpay": self._validate_worldpay,
            "dama": self._validate_dama,
        }
        if handler := _provider_handlers.get(self.provider):
            result = await handler(clean_number)
            result["brand"] = brand
            return result

        return {"valid": True, "brand": brand, "info": "Validated locally (Luhn pass)"}

    async def _validate_cardio(self, _card_number: str) -> dict[str, Any]:
        """Card.io validation (stub)."""
        await asyncio.sleep(0)
        return {
            "valid": True,
            "provider": "cardio",
            "info": "Card.io implementation pending",
        }

    async def _validate_dama(self, _card_number: str) -> dict[str, Any]:
        """DAMA validation (stub)."""
        await asyncio.sleep(0)
        return {
            "valid": True,
            "provider": "dama",
            "info": "DAMA implementation pending",
        }

    async def _validate_worldpay(self, _card_number: str) -> dict[str, Any]:
        """Worldpay validation (stub)."""
        await asyncio.sleep(0)
        return {
            "valid": True,
            "provider": "worldpay",
            "info": "Worldpay implementation pending",
        }

    async def _validate_mock(self, card_number: str) -> dict[str, Any]:
        """Mock validator for testing; accepts numbers ending in '0005'."""
        await asyncio.sleep(0)
        is_valid = card_number.endswith("0005")
        return {
            "valid": is_valid,
            "provider": "mock",
            "info": "Mock validation pass" if is_valid else "Rejected by mock",
        }

    async def _validate_numbrify(self, card_number: str) -> dict[str, Any]:
        """Numbrify (APILayer) BIN validation."""
        if not self.api_key:
            logger.warning("Numbrify API Key missing; falling back to local validation")
            return {"valid": True, "provider": "numbrify", "source": "local_fallback"}

        url = "https://api.apilayer.com/bincheck/check"
        headers = {"apikey": self.api_key}
        params = {"bin": card_number[:6]}

        try:
            response = await self.client.get(url, headers=headers, params=params)
            if response.status_code == 200:
                return {
                    "valid": True,
                    "provider": "numbrify",
                    "external_info": response.json(),
                }
            if response.status_code == 401:
                logger.error("Numbrify API key rejected (401)")
            if response.status_code == 429:
                logger.warning("Numbrify rate limited (429)")
        except httpx.HTTPError as e:
            logger.error("Numbrify validation failed: %s", e)

        return {
            "valid": True,
            "provider": "numbrify",
            "error": "External call failed, trusted local check",
        }

    async def _validate_binlist(self, card_number: str) -> dict[str, Any]:
        """Free BIN lookup via binlist.net (no key required for low volume)."""
        url = f"https://lookup.binlist.net/{card_number[:8]}"
        try:
            response = await self.client.get(url, headers={"Accept-Version": "3"})
            if response.status_code == 200:
                return {
                    "valid": True,
                    "provider": "binlist",
                    "external_info": response.json(),
                }
            if response.status_code == 429:
                logger.warning("BinList rate limited (429)")
        except httpx.HTTPError as e:
            logger.debug("BinList lookup failed: %s", e)

        return {"valid": True, "provider": "binlist", "source": "local_only"}

    async def close(self) -> None:
        await self.client.aclose()
