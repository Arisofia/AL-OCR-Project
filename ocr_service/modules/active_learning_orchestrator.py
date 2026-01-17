"""
Active Learning Orchestrator for AL-OCR-Project.
Automates the end-to-end flow: Ingestion -> Sampling -> Validation -> Drift Detection.
"""

import asyncio
import logging
from typing import Any, Optional, cast

import numpy as np
import pandas as pd

from ocr_service.config import get_settings
from ocr_service.modules.active_learning import HybridSampling, OCRModel
from ocr_service.modules.learning_engine import LearningEngine
from ocr_service.utils.drift_detection import check_for_drift
from ocr_service.utils.validation import validate_ocr_batch

logger = logging.getLogger("ocr-service.al-orchestrator")


class ALOrchestrator:
    """
    Automates the Active Learning lifecycle.
    """

    def __init__(
        self, learning_engine: LearningEngine, model: Optional[OCRModel] = None
    ):
        self.learning_engine = learning_engine
        self.settings = get_settings()
        self.model = model
        self.sampling_strategy = HybridSampling(
            n_clusters=self.settings.al_n_clusters, diversity_ratio=0.3
        )

    async def run_cycle(self, n_samples: Optional[int] = None) -> dict[str, Any]:
        """
        Executes one full Active Learning cycle.
        """
        actual_n_samples = n_samples or self.settings.al_cycle_samples
        logger.info("Starting Active Learning cycle...")

        if self.model is None:
            logger.error("AL cycle aborted: No OCR model provided.")
            return {"status": "error", "message": "No model provided"}

        # 1. Fetch recent results from LearningEngine (Supabase)
        recent_data = await self._fetch_recent_results(limit=200)
        if not recent_data or len(recent_data) < actual_n_samples:
            logger.warning(
                "Insufficient data to run AL cycle. Found: %d", len(recent_data)
            )
            return {"status": "insufficient_data", "count": len(recent_data)}

        df = pd.DataFrame(recent_data)

        # 2. Data Quality Gate: Validate the batch
        # We need columns: [image_path, ocr_text, confidence, user_label]
        # In our case, Supabase stores font_metadata and accuracy_score.
        # We'll map them to the expected schema for validation.
        validation_df = self._prepare_for_validation(df)
        if not validate_ocr_batch(validation_df):
            logger.error("AL cycle aborted: Data quality gate failed.")
            return {"status": "validation_failed"}

        # 3. Hybrid Sampling: Select candidates for labeling
        # In a real scenario, this would be a loaded PyTorch/TF model.
        selected_indices = self.sampling_strategy.select_indices(
            self.model,
            np.random.rand(len(df), 128),  # Mock embeddings for now
            actual_n_samples,
        )
        candidates = df.iloc[selected_indices]
        logger.info(
            "Selected %d candidates for labeling via Hybrid Strategy.", len(candidates)
        )

        # 4. Drift Detection: Compare current batch vs historical baseline
        try:
            reference_df = pd.read_csv(self.settings.reference_baseline_path)
            drift_detected = check_for_drift(reference_df, validation_df)
            if drift_detected:
                logger.warning("AL cycle alert: Model drift detected!")
        except FileNotFoundError:
            logger.info("Reference baseline not found. Skipping drift detection.")
            drift_detected = False

        return {
            "status": "success",
            "samples_selected": len(candidates),
            "drift_detected": drift_detected,
            "candidates": candidates.to_dict(orient="records")[
                :5
            ],  # Return top 5 for preview
        }

    async def _fetch_recent_results(self, limit: int = 100) -> list[dict[str, Any]]:
        """Retrieves data prioritizing Supabase with Local Fallback."""
        # Try Cloud
        if self.learning_engine.client:
            try:

                def _fetch():
                    if self.learning_engine.client:
                        return (
                            self.learning_engine.client.table("learning_patterns")
                            .select("*")
                            .order("created_at", desc=True)
                            .limit(limit)
                            .execute()
                        )
                    return None

                res = await asyncio.to_thread(_fetch)
                if res and res.data:
                    return cast(list[dict[str, Any]], res.data)
            except Exception:
                logger.info("Cloud fetch failed in orchestrator, using local fallback")

        # Fallback to Local Engine logic
        def _local():
            return self.learning_engine._load_patterns()[-limit:]

        return await asyncio.to_thread(_local)

    def _prepare_for_validation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Maps Supabase schema to Validation Gate schema."""
        # Schema: [id, doc_type, font_metadata, accuracy_score, created_at, version]
        v_df = pd.DataFrame()
        ids = df.get("id")
        if ids is not None and hasattr(ids, "apply"):
            v_df["image_path"] = ids.apply(lambda x: f"s3://bucket/{x}.png")
        else:
            v_df["image_path"] = "unknown"

        v_df["ocr_text"] = "mock_text"  # In real AL, we'd store the actual text
        v_df["confidence"] = df.get("accuracy_score", 0.0)
        v_df["user_label"] = "pending"
        return v_df
