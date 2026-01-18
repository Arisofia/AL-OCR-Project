"""
Learning engine module for continuous improvement of OCR results.
Hybrid implementation: Prioritizes Supabase (Cloud) with an automatic
Local JSON fallback for resilience and Free Tier management.
"""

import asyncio
import json
import logging
import os
import time
from typing import Any, Optional, cast

from supabase import Client, create_client

from ocr_service.config import get_settings

logger = logging.getLogger("ocr-service.learning")


class LearningEngine:
    """
    Implements continuous learning. Orchestrates data between Supabase
    and Local Storage to ensure build integrity and stability.
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.storage_path = self.settings.local_data_path
        self.client: Optional[Client] = None
        self._knowledge_cache: dict[str, Any] = {}

        # Initialize Supabase with short timeout for Free Tier resilience
        if self.settings.supabase_url and self.settings.supabase_service_role:
            try:
                # Use a specific configuration if available in newer versions,
                # or just handle the flow gracefully
                self.client = create_client(
                    self.settings.supabase_url, self.settings.supabase_service_role
                )
                logger.info("Supabase integration active (Free Tier Mode)")
            except Exception as e:
                logger.error("Supabase init bypassed, local-only mode: %s", e)

        # Ensure local directory exists
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)

    def check_connection(self) -> bool:
        """Verifies if at least one storage method is healthy."""
        if self.client:
            try:
                self.client.table("learning_patterns").select("count").limit(
                    1
                ).execute()
                return True
            except Exception:
                logger.warning("Supabase health check failed")

        return os.access(os.path.dirname(self.storage_path), os.W_OK)

    async def learn_from_result(
        self, doc_type: str, font_meta: dict[str, Any], accuracy_score: float
    ) -> None:
        """Persists learning data to Supabase and Local simultaneously."""
        data = {
            "doc_type": doc_type,
            "font_metadata": font_meta,
            "accuracy_score": accuracy_score,
            "version": self.settings.version,
            "created_at": time.time(),
        }

        # 1. Background task for Local Persistence (Always)
        save_task = asyncio.create_task(self._save_local(data))

        # 2. Attempt Supabase Upsert (If active)
        if self.client:
            try:

                def _upsert() -> None:
                    if self.client:
                        # Use type ignore as Supabase's JSON type is slightly
                        # incompatible with inferred dict[str, object]
                        self.client.table("learning_patterns").upsert(
                            data  # type: ignore[arg-type]
                        ).execute()

                await asyncio.to_thread(_upsert)
                logger.debug("Cloud sync successful")
            except Exception as e:
                logger.warning("Cloud sync failed (possibly quota exceeded): %s", e)

        await save_task

    async def get_pattern_knowledge(self, doc_type: str) -> Optional[dict[str, Any]]:
        """Retrieves best pattern, prioritizing Supabase then Local."""
        # Try Cloud first
        if self.client:
            try:

                def _fetch() -> Optional[Any]:
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

                res = await asyncio.to_thread(_fetch)
                if res and res.data:
                    return cast(dict[str, Any], res.data[0])
            except Exception:
                logger.info("Cloud fetch failed, searching local cache")

        # Fallback to Local
        return await self._fetch_local(doc_type)

    async def _save_local(self, data: dict[str, Any]) -> None:
        def _io() -> None:
            patterns = self._load_patterns()
            patterns.append(data)
            if len(patterns) > 500:
                patterns = patterns[-500:]
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(patterns, f, indent=2)

        await asyncio.to_thread(_io)

    async def _fetch_local(self, doc_type: str) -> Optional[dict[str, Any]]:
        def _io() -> Optional[dict[str, Any]]:
            patterns = self._load_patterns()
            matches = [p for p in patterns if p.get("doc_type") == doc_type]
            if not matches:
                return None
            matches.sort(key=lambda x: x.get("accuracy_score", 0.0), reverse=True)
            return matches[0]

        return await asyncio.to_thread(_io)

    def _load_patterns(self) -> list[dict[str, Any]]:
        if not os.path.exists(self.storage_path):
            return []
        try:
            with open(self.storage_path, encoding="utf-8") as f:
                return cast(list[dict[str, Any]], json.load(f))
        except Exception:
            return []
