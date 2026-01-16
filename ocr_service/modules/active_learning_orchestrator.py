"""
Active Learning Orchestrator for AL-OCR-Project.
Automates the end-to-end flow: Ingestion -> Sampling -> Validation -> Drift Detection.
"""

import asyncio
import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from ocr_service.modules.active_learning import HybridSampling
from ocr_service.modules.learning_engine import LearningEngine
from ocr_service.utils.validation import validate_ocr_batch
from ocr_service.utils.drift_detection import check_for_drift

logger = logging.getLogger("ocr-service.al-orchestrator")


class ALOrchestrator:
    """
    Automates the Active Learning lifecycle.
    """

    def __init__(self, learning_engine: LearningEngine):
        self.learning_engine = learning_engine
        self.sampling_strategy = HybridSampling(n_clusters=5, diversity_ratio=0.3)

    async def run_cycle(self, n_samples: int = 50) -> Dict[str, Any]:
        """
        Executes one full Active Learning cycle.
        """
        logger.info("Starting Active Learning cycle...")

        # 1. Fetch recent results from LearningEngine (Supabase)
        recent_data = await self._fetch_recent_results(limit=200)
        if not recent_data or len(recent_data) < n_samples:
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
        # For simplicity in this automation script, we'll mock the model component
        # In a real scenario, this would be a loaded PyTorch/TF model.
        mock_model = MockOCRModel()
        selected_indices = self.sampling_strategy.select_indices(
            mock_model,
            np.random.rand(len(df), 128),  # Mock embeddings
            n_samples,
        )
        candidates = df.iloc[selected_indices]
        logger.info(
            "Selected %d candidates for labeling via Hybrid Strategy.", len(candidates)
        )

        # 4. Drift Detection: Compare current batch vs historical baseline
        # (Assuming we have a reference file)
        try:
            reference_df = pd.read_csv("data/reference_baseline.csv")
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

    async def _fetch_recent_results(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Retrieves data from Supabase."""
        if not self.learning_engine.client:
            return []

        def _fetch():
            return (
                self.learning_engine.client.table("learning_patterns")
                .select("*")
                .order("created_at", desc=True)
                .limit(limit)
                .execute()
            )

        result = await asyncio.to_thread(_fetch)
        if not result:
            return []
        return result.data if result.data else []

    def _prepare_for_validation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Maps Supabase schema to Validation Gate schema."""
        # Schema: [id, doc_type, font_metadata, accuracy_score, created_at, version]
        v_df = pd.DataFrame()
        v_df["image_path"] = df.get("id", "unknown").apply(
            lambda x: f"s3://bucket/{x}.png"
        )
        v_df["ocr_text"] = "mock_text"  # In real AL, we'd store the actual text
        v_df["confidence"] = df.get("accuracy_score", 0.0)
        v_df["user_label"] = "pending"
        return v_df


class MockOCRModel:
    """Mock model to satisfy the Protocol during automation demo."""

    def get_embeddings(self, data: np.ndarray) -> np.ndarray:
        return np.random.rand(len(data), 128)

    def predict_proba(self, data: np.ndarray) -> np.ndarray:
        # Return mock high uncertainty for first half, low for second
        n = len(data)
        probs = np.zeros((n, 2))
        probs[:, 0] = 0.5  # High uncertainty
        probs[:, 1] = 0.5
        return probs
