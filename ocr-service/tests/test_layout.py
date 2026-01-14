import numpy as np
import cv2
from modules.layout import DocumentLayoutAnalyzer


def test_detect_regions_empty():
    # Test with empty bytes
    regions = DocumentLayoutAnalyzer.detect_regions(b"")
    assert regions == []


def test_detect_regions_real_image():
    # Create a dummy image with two white blocks on black background
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.rectangle(img, (10, 10), (30, 30), (255, 255, 255), -1)
    cv2.rectangle(img, (50, 50), (80, 80), (255, 255, 255), -1)

    _, img_bytes = cv2.imencode(".jpg", img)

    regions = DocumentLayoutAnalyzer.detect_regions(img_bytes.tobytes())

    # It should find at least 2 regions (depending on dilation)
    assert len(regions) >= 2
    for r in regions:
        assert "bbox" in r
        assert "rel_bbox" in r
        assert "area_ratio" in r


def test_classify_layout():
    regions = [{"area_ratio": 0.01}] * 25
    assert DocumentLayoutAnalyzer.classify_layout(regions) == "dense_text"

    regions = [{"area_ratio": 0.5}]
    assert DocumentLayoutAnalyzer.classify_layout(regions) == "large_blocks"

    assert DocumentLayoutAnalyzer.classify_layout([]) == "empty"
