from abc import ABC, abstractmethod

import numpy as np


class OCRModel(ABC):
    @abstractmethod
    def predict_proba(self, images: list[np.ndarray]) -> np.ndarray:
        pass

    @abstractmethod
    def train(self, labeled_data: list):
        pass


class QueryStrategy(ABC):
    @abstractmethod
    def select_indices(
        self, model: OCRModel, unlabeled_pool: list[np.ndarray], n_samples: int
    ) -> list[int]:
        pass


class UncertaintySampling(QueryStrategy):
    def select_indices(
        self, model: OCRModel, unlabeled_pool: list[np.ndarray], n_samples: int
    ) -> list[int]:
        probs = model.predict_proba(unlabeled_pool)
        max_probs = np.max(probs, axis=2)
        sequence_confidence = np.min(max_probs, axis=1)
        uncertainty_scores = 1 - sequence_confidence
        selected_indices = np.argsort(uncertainty_scores)[-n_samples:]
        return list(map(int, selected_indices.tolist()))


class ActiveLearningLoop:
    def __init__(self, model: OCRModel, strategy: QueryStrategy):
        self.model = model
        self.strategy = strategy

    def run_cycle(self, unlabeled_pool, n_samples):
        self.strategy.select_indices(self.model, unlabeled_pool, n_samples)
