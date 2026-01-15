from abc import ABC, abstractmethod
from typing import List, Dict, Any
import numpy as np

class OCRBaseModel(ABC):
    """
    The contract that any OCR model must fulfill to work with the AL Loop.
    """
    @abstractmethod
    def load(self, weights_path: str):
        """Load model weights from disk."""
        pass

    @abstractmethod
    def predict(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Return text and metadata.
        Must return: {'text': str, 'boxes': list}
        """
        pass

    @abstractmethod
    def predict_proba(self, image: np.ndarray) -> np.ndarray:
        """
        CRITICAL: Must return probability distribution for Active Learning.
        Returns: Confidence score (0.0 - 1.0) or full logits.
        """
        pass
    
    @abstractmethod
    def get_embeddings(self, image: np.ndarray) -> np.ndarray:
        """
        Required for Diversity Sampling (Clustering).
        Returns: Feature vector (e.g., 512-dim float array).
        """
        pass
