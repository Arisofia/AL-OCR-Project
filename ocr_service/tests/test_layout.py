"""Tests for layout region detection and classification."""

import cv2
import numpy as np

from ocr_service.modules.layout import DocumentLayoutAnalyzer


def test_detect_regions_empty():
    regions = DocumentLayoutAnalyzer.detect_regions(b"")
    assert not regions


def test_detect_regions_real_image():
    """Test layout detection on a synthetic image."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.rectangle(img, (10, 10), (30, 30), (255, 255, 255), -1)
    cv2.rectangle(img, (50, 50), (80, 80), (255, 255, 255), -1)

    _, img_bytes = cv2.imencode(".jpg", img)

    regions = DocumentLayoutAnalyzer.detect_regions(img_bytes.tobytes())

    assert len(regions) >= 2
    assert all("bbox" in r for r in regions)
    assert all("rel_bbox" in r for r in regions)
    assert all("area_ratio" in r for r in regions)


def test_classify_layout():
    """Test layout classification logic."""
    regions = [{"area_ratio": 0.01}] * 25
    assert DocumentLayoutAnalyzer.classify_layout(regions) == "dense_text"

    regions = [{"area_ratio": 0.5}]
    assert DocumentLayoutAnalyzer.classify_layout(regions) == "large_blocks"

    assert DocumentLayoutAnalyzer.classify_layout([]) == "empty"
