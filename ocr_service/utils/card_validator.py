import logging
import httpx
import os
from typing import Optional, List, Dict, Any
from creditcard import CreditCard

logger = logging.getLogger("ocr-service.card-validator")

class CardValidator:
    """
    Client for external credit card validation APIs to verify predicted numbers.
    Supports providers like NumVerify (Numbrify), Worldpay, etc.
    Utilizes local 'creditcard' package for initial validation.
    """
    
    def __init__(self, provider: Optional[str] = None, api_key: Optional[str] = None):
        # Default to environment variables if not provided
        self.provider = provider or os.getenv("CARD_VALIDATOR_PROVIDER", "binlist")
        self.api_key = api_key or os.getenv("CARD_VALIDATOR_API_KEY")
        self.client = httpx.AsyncClient(timeout=5.0)

    async def validate(self, card_number: str) -> Dict[str, Any]:
        """
        Validate a full 16-digit card number.
        First performs local validation using 'creditcard' package,
        then attempts external verification if a provider is configured.
        """
        clean_number = card_number.replace(" ", "")
        if not clean_number or len(clean_number) < 13:
            return {"valid": False, "error": "Invalid length"}

        # 1. Local Validation (Luhn + Brand identification)
        local_card = CreditCard(clean_number)
        if not local_card.is_valid():
            return {"valid": False, "error": "Local Luhn check failed"}

        brand = local_card.get_brand()
        
        # 2. External Validation (if configured)
        if self.provider == "mock":
            return await self._validate_mock(clean_number)
        elif self.provider == "numbrify":
            external = await self._validate_numbrify(clean_number)
            external["brand"] = brand
            return external
        elif self.provider == "binlist":
            external = await self._validate_binlist(clean_number)
            external["brand"] = brand
            return external
        elif self.provider == "cardio":
            return await self._validate_cardio(clean_number)
        elif self.provider == "worldpay":
            return await self._validate_worldpay(clean_number)
        elif self.provider == "dama":
            return await self._validate_dama(clean_number)
        
        # Default to local validation success if no external provider matched
        return {
            "valid": True, 
            "brand": brand,
            "info": "Validated locally (Luhn pass)"
        }

    async def _validate_cardio(self, card_number: str) -> Dict[str, Any]:
        """Card.io validation (stub)."""
        return {"valid": True, "info": "Card.io implementation pending"}

    async def _validate_dama(self, card_number: str) -> Dict[str, Any]:
        """DAMA validation (stub)."""
        return {"valid": True, "info": "DAMA implementation pending"}

    async def _validate_mock(self, card_number: str) -> Dict[str, Any]:
        """Mock validator for testing; accepts numbers ending in '0005'."""
        is_valid = card_number.endswith("0005")
        return {
            "valid": is_valid,
            "provider": "mock",
            "info": "Mock validation pass" if is_valid else "Rejected by mock"
        }

    async def _validate_numbrify(self, card_number: str) -> Dict[str, Any]:
        """Numbrify (APILayer) BIN validation."""
        if not self.api_key:
            logger.warning("Numbrify API Key missing; falling back to local validation")
            return {"valid": True, "source": "local_fallback"}
        
        try:
            # Numbrify usually refers to APILayer's bincheck
            url = f"https://api.apilayer.com/bincheck/check"
            headers = {"apikey": self.api_key}
            params = {"bin": card_number[:6]}
            
            response = await self.client.get(url, headers=headers, params=params)
            if response.status_code == 200:
                data = response.json()
                return {"valid": True, "external_info": data}
        except Exception as e:
            logger.error("Numbrify validation failed: %s", e)
        
        return {"valid": True, "error": "External call failed, trusted local check"}

    async def _validate_binlist(self, card_number: str) -> Dict[str, Any]:
        """Free BIN lookup via binlist.net (no key required for low volume)."""
        try:
            url = f"https://lookup.binlist.net/{card_number[:8]}"
            response = await self.client.get(url, headers={"Accept-Version": "3"})
            if response.status_code == 200:
                return {"valid": True, "external_info": response.json()}
            elif response.status_code == 429:
                logger.warning("BinList rate limited")
        except Exception as e:
            logger.debug("BinList lookup failed: %s", e)
        
        return {"valid": True, "source": "local_only"}

    async def _validate_worldpay(self, card_number: str) -> Dict[str, Any]:
        """Worldpay validation (stub)."""
        return {"valid": True, "info": "Worldpay implementation pending"}

    async def close(self):
        await self.client.aclose()

