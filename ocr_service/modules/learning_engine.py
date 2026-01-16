"""
Learning engine module for continuous improvement of OCR results.

This module provides functionality to store and retrieve document patterns
using Supabase, enabling the service to learn from historical data.
"""

import asyncio
import logging
import time
from typing import Any, Dict, Optional

from supabase import Client, create_client

from ocr_service.config import get_settings

logger = logging.getLogger("ocr-service.learning")


class LearningEngine:
    """
    Implements continuous learning by storing and retrieving document patterns,
    font characteristics, and correction history in Supabase.
    """

    def __init__(self) -> None:
        """
        Initializes the LearningEngine and Supabase client.
        """
        self.settings = get_settings()
        self.client: Optional[Client] = None
        self._knowledge_cache: Dict[str, Any] = {}
        self._last_check_time = 0.0
        self._last_check_result = False

        if self.settings.supabase_url and self.settings.supabase_service_role:
            try:
                self.client = create_client(
                    self.settings.supabase_url, self.settings.supabase_service_role
                )
            except Exception as e:  # pylint: disable=broad-exception-caught
                logger.error("Failed to initialize Supabase client: %s", e)

    def check_connection(self) -> bool:
        """
        Validates Supabase connectivity by performing a simple query.
        Results are cached for 60 seconds to avoid redundant API calls.
        """
        if not self.client:
            return False

        # Return cached result if fresh
        if time.time() - self._last_check_time < 60:
            return self._last_check_result

        try:
            # Simple heartbeat query
            self.client.table("learning_patterns").select("count").limit(1).execute()
            self._last_check_result = True
        except Exception as e:
            logger.warning("Supabase health check failed: %s", e)
            self._last_check_result = False

        self._last_check_time = time.time()
        return self._last_check_result

    async def learn_from_result(
        self, doc_type: str, font_meta: Dict[str, Any], accuracy_score: float
    ) -> None:
        """
        Records document characteristics for future reference and pattern matching.
        """
        if not self.client:
            return

        data = {
            "doc_type": doc_type,
            "font_metadata": font_meta,
            "accuracy_score": accuracy_score,
            "version": self.settings.version,
        }

        try:
            # Clear cache for this doc_type to ensure freshness next time
            self._knowledge_cache.pop(doc_type, None)

            # Upsert into 'learning_patterns' in a background thread
            def _upsert() -> None:
                if self.client:
                    table = self.client.table("learning_patterns")
                    table.upsert(data).execute()  # type: ignore

            await asyncio.to_thread(_upsert)
        except Exception as e:
            logger.warning("Failed to record learning data: %s", e)

    async def get_pattern_knowledge(self, doc_type: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves previously learned patterns for a specific document type.
        """
        if not self.client:
            return None

        # Check cache first
        if doc_type in self._knowledge_cache:
            logger.debug("Knowledge cache hit for %s", doc_type)
            from typing import cast

            return cast(Optional[Dict[str, Any]], self._knowledge_cache[doc_type])

        try:

            def _fetch() -> Any:
                if self.client:
                    return (
                        self.client.table("learning_patterns")
                        .select("*")
                        .eq("doc_type", doc_type)
                        .order("accuracy_score", desc=True)
                        .limit(1)
                        .execute()
                    )
                return None

            result = await asyncio.to_thread(_fetch)
            if not result:
                return None
            knowledge = result.data[0] if result.data else None
            if knowledge:
                self._knowledge_cache[doc_type] = knowledge
            from typing import cast

            return cast(Optional[Dict[str, Any]], knowledge)
        except Exception as e:
            logger.error("Failed to fetch pattern knowledge: %s", e)
            return None
