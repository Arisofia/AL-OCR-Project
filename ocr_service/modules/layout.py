"""
Module for document layout analysis using computer vision.
Used to identify regions of interest and classify document structure.
"""

import logging
from typing import Any, cast

import cv2
import numpy as np

__all__ = ["DocumentLayoutAnalyzer"]

logger = logging.getLogger("ocr-service.layout")


class DocumentLayoutAnalyzer:
    """
    Analyzes the physical layout of a document to identify Regions of Interest (ROI).
    Helps in focused OCR and understanding document structure.
    """

    @staticmethod
    def detect_regions(image_bytes: bytes) -> list[dict[str, Any]]:
        """
        Detects bounding boxes of text/content regions in the image.
        Adds robust error handling.
        """
        if not image_bytes:
            logger.warning("No image bytes provided for region detection.")
            return []
        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                logger.warning("Failed to decode image for layout analysis.")
                return []
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            white_pixels = cv2.countNonZero(thresh)
            img_area = img.shape[0] * img.shape[1]
            if white_pixels > (img_area / 2):
                thresh = cv2.bitwise_not(thresh)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            dilated = cv2.dilate(thresh, kernel, iterations=3)
            contours, _ = cv2.findContours(
                dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            regions = []
            img_h, img_w = img.shape[:2]
            for i, cnt in enumerate(contours):
                x, y, w, h = cv2.boundingRect(cnt)
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
            regions.sort(key=lambda r: cast(list[int], r["bbox"])[1])
            logger.debug("Detected %d regions in document.", len(regions))
            return regions
        except Exception as e:
            logger.error("Exception in detect_regions: %s", e)
            return []

    @staticmethod
    def classify_layout(regions: list[dict[str, Any]]) -> str:
        """
        Simple heuristic classification of the document layout.
        Adds robust error handling.
        """
        try:
            if not regions:
                logger.info("No regions provided for layout classification.")
                return "empty"
            num_regions = len(regions)
            avg_area = sum(r["area_ratio"] for r in regions) / num_regions
            if num_regions > 20 and avg_area < 0.05:
                return "dense_text"  # Like a page of a book
            if num_regions < 10 and any(r["area_ratio"] > 0.4 for r in regions):
                return "large_blocks"  # Like a poster or simple form
            return "standard_form"
        except Exception as e:
            logger.error("Exception in classify_layout: %s", e)
            return "error"
