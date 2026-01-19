"""
Logic for calculating OCR confidence scores based on text analysis.
"""

import re
from typing import Optional

__all__ = ["ConfidenceScorer"]


class ConfidenceScorer:
    """
    Evaluates OCR quality based on linguistic patterns,
    character density, and document markers.
    """

    def __init__(self, markers: Optional[list[str]] = None):
        """
        Initializes the scorer with a set of document markers.
        """
        self.markers = markers or [
            "date",
            "fecha",
            "total",
            "invoice",
            "factura",
            "name",
            "nombre",
            "id",
            "dni",
            "tax",
            "iva",
        ]

    def calculate(self, text: str) -> float:
        """
        Calculates a confidence score between 0.0 and 1.0 for the given text.
        """
        if not text or len(text.strip()) == 0:
            return 0.0

        # 1. Alphanumeric density
        alnum_count = sum(1 for c in text if c.isalnum())
        density = alnum_count / len(text)

        # 2. Word-like patterns (sequences of letters)
        words = re.findall(r"\b[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ]{2,}\b", text)
        word_count = len(words)

        # 3. Document marker detection
        marker_score = sum(0.05 for m in self.markers if m in text.lower())
        marker_score = min(0.2, marker_score)

        # 4. Length factor
        length_factor = min(1.0, len(text) / 100)

        # 5. Word quality factor
        word_factor = min(1.0, word_count / 10) if word_count > 0 else 0

        # Combine metrics: 40% density, 40% word quality, 20% markers
        base_score = (density * 0.4) + (word_factor * 0.4) + marker_score

        return round(base_score * length_factor, 2)
