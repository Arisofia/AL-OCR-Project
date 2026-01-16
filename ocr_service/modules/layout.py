"""
Module for document layout analysis using computer vision.
Used to identify regions of interest and classify document structure.
"""

import logging
from typing import Any, Dict, List, cast

import cv2
import numpy as np

logger = logging.getLogger("ocr-service.layout")


class DocumentLayoutAnalyzer:
    """
    Analyzes the physical layout of a document to identify Regions of Interest (ROI).
    Helps in focused OCR and understanding document structure.
    """

    @staticmethod
    def detect_regions(image_bytes: bytes) -> List[Dict[str, Any]]:
        """
        Detects bounding boxes of text/content regions in the image.
        """
        if not image_bytes:
            return []

        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            logger.warning("Failed to decode image for layout analysis.")
            return []

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Determine if we should invert (assume background is the most frequent color)
        # For simplicity, we can use Otsu and then check if we need to invert
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # If more than half the image is white the background is white.
        # In that case invert so text is white on a dark background for contours
        white_pixels = cv2.countNonZero(thresh)
        img_area = img.shape[0] * img.shape[1]
        if white_pixels > (img_area / 2):
            thresh = cv2.bitwise_not(thresh)

        # Use a smaller kernel for dilation to avoid merging distant blocks
        # (helps keep separate regions in tests)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilated = cv2.dilate(thresh, kernel, iterations=3)

        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        regions = []
        img_h, img_w = img.shape[:2]

        for i, cnt in enumerate(contours):
            x, y, w, h = cv2.boundingRect(cnt)

            # Filter out very small noise
            if w < 20 or h < 10:
                continue

            regions.append(
                {
                    "id": i,
                    "bbox": [x, y, w, h],
                    "rel_bbox": [x / img_w, y / img_h, w / img_w, h / img_h],
                    "area_ratio": (w * h) / (img_w * img_h),
                }
            )

        # Sort regions by y position (top to bottom)
        regions.sort(key=lambda r: cast(List[int], r["bbox"])[1])
        logger.debug("Detected %d regions in document.", len(regions))
        return regions

    @staticmethod
    def classify_layout(regions: List[Dict[str, Any]]) -> str:
        """
        Simple heuristic classification of the document layout.
        """
        if not regions:
            return "empty"

        num_regions = len(regions)
        avg_area = sum(r["area_ratio"] for r in regions) / num_regions

        if num_regions > 20 and avg_area < 0.05:
            return "dense_text"  # Like a page of a book
        if num_regions < 10 and any(r["area_ratio"] > 0.4 for r in regions):
            return "large_blocks"  # Like a poster or simple form
        return "standard_form"
