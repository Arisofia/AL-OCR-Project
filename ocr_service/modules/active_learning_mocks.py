"""
Mocks for Active Learning components to be used in tests or demos.
"""

import numpy as np


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
