"""
Active Learning strategies for document intelligence.
Implements hybrid query strategies to mitigate sampling bias.
"""

import numpy as np
from sklearn.cluster import KMeans
from typing import List, Protocol, runtime_checkable


@runtime_checkable
class OCRModel(Protocol):
    """Protocol for models that can provide embeddings and predictions."""

    def get_embeddings(self, data: np.ndarray) -> np.ndarray:
        ...

    def predict_proba(self, data: np.ndarray) -> np.ndarray:
        ...


class QueryStrategy:
    """Base class for active learning query strategies."""

    def select_indices(
        self, model: OCRModel, unlabeled_data: np.ndarray, n_samples: int
    ) -> List[int]:
        raise NotImplementedError


class HybridSampling(QueryStrategy):
    """
    Implements a hybrid strategy:
    1. Clusters the unlabeled data to find diverse groups.
    2. Selects the most uncertain samples from EACH cluster.

    Mitigates 'Sampling Bias' and 'Outlier Flooding'.
    """

    def __init__(self, n_clusters: int = 10, diversity_ratio: float = 0.3):
        self.n_clusters = n_clusters
        self.diversity_ratio = diversity_ratio  # 30% diversity, 70% uncertainty

    def select_indices(
        self, model: OCRModel, unlabeled_data: np.ndarray, n_samples: int
    ) -> List[int]:
        # 1. Get Embeddings (Feature Vectors) from the model backbone
        embeddings = model.get_embeddings(unlabeled_data)

        # 2. Get Uncertainty Scores (Least Confidence)
        probs = model.predict_proba(unlabeled_data)
        uncertainty = 1 - np.max(probs, axis=1)

        # --- Diversity Step: K-Means Clustering ---
        actual_n_clusters = min(self.n_clusters, len(unlabeled_data))
        if actual_n_clusters < 1:
            return []

        kmeans = KMeans(n_clusters=actual_n_clusters, random_state=42, n_init="auto")
        cluster_labels = kmeans.fit_predict(embeddings)

        selected_indices: List[int] = []
        samples_per_cluster = max(1, n_samples // actual_n_clusters)

        # Select samples from each cluster to ensure coverage
        for i in range(actual_n_clusters):
            # Get indices belonging to this cluster
            cluster_indices = np.where(cluster_labels == i)[0]

            if len(cluster_indices) == 0:
                continue

            # Within this cluster, pick the most UNCERTAIN ones
            cluster_uncertainties = uncertainty[cluster_indices]

            # Sort by uncertainty (descending)
            sorted_idx_local = np.argsort(cluster_uncertainties)[::-1]

            # Select top K from this cluster
            top_k_local = sorted_idx_local[:samples_per_cluster]
            selected_indices.extend(cluster_indices[top_k_local].tolist())

        # Handle remainder or over-sampling
        if len(selected_indices) > n_samples:
            selected_indices = selected_indices[:n_samples]
        elif len(selected_indices) < n_samples:
            needed = n_samples - len(selected_indices)
            remaining_mask = np.ones(len(uncertainty), dtype=bool)
            remaining_mask[selected_indices] = False
            remaining_indices = np.where(remaining_mask)[0]

            if len(remaining_indices) > 0:
                extra_needed = min(needed, len(remaining_indices))
                extra_local = np.argsort(uncertainty[remaining_indices])[::-1][
                    :extra_needed
                ]
                selected_indices.extend(remaining_indices[extra_local].tolist())

        return selected_indices
