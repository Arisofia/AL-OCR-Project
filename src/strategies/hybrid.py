import numpy as np
from sklearn.cluster import KMeans
from src.core.model_interface import OCRBaseModel
from typing import List

class HybridSampler:
    def __init__(self, n_clusters: int = 10, diversity_ratio: float = 0.3):
        self.n_clusters = n_clusters
        self.diversity_ratio = diversity_ratio

    def select_batch(self, model: OCRBaseModel, unlabeled_images: List[np.ndarray], n_samples: int) -> List[int]:
        """
        Selects indices of images to label using Uncertainty + Clustering.
        """
        confidences = np.array([model.predict_proba(img) for img in unlabeled_images])
        embeddings = np.array([model.get_embeddings(img) for img in unlabeled_images])
        uncertainty_scores = 1.0 - confidences
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(embeddings)
        selected_indices = []
        samples_per_cluster = max(1, n_samples // self.n_clusters)
        for i in range(self.n_clusters):
            cluster_idx = np.where(cluster_labels == i)[0]
            if len(cluster_idx) == 0:
                continue
            cluster_uncertainties = uncertainty_scores[cluster_idx]
            sorted_local_idx = np.argsort(cluster_uncertainties)[::-1]
            top_k = sorted_local_idx[:samples_per_cluster]
            selected_indices.extend(cluster_idx[top_k])
        return selected_indices
