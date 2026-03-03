#!/usr/bin/env python3
"""
Rescan all 6 hidden positions using corrected coordinates from column analysis.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Final, Iterable

import cv2
import numpy as np
import pytesseract

IMAGE_PATH: Final = "/Users/jenineferderas/Desktop/card_image.jpg"
TESS_NUMERIC_ONLY: Final = "-c tessedit_char_whitelist=0123456789"
CFG_SINGLE_DIGIT: Final = f"--oem 3 --psm 13 {TESS_NUMERIC_ONLY}"
CFG_GROUP: Final = f"--oem 3 --psm 7 {TESS_NUMERIC_ONLY}"
SINGLE_DIGIT_CONFIGS: Final[tuple[str, ...]] = (
    f"--oem 3 --psm 10 {TESS_NUMERIC_ONLY}",
    CFG_SINGLE_DIGIT,
    f"--oem 3 --psm 8 {TESS_NUMERIC_ONLY}",
)
PAIR_CONFIGS: Final[tuple[str, ...]] = (CFG_GROUP, CFG_SINGLE_DIGIT)
OCR_EXCEPTIONS = (pytesseract.TesseractError, RuntimeError, TypeError, ValueError)

CENTERS: Final[dict[int, int]] = {
    6: 647,
    7: 708,
    8: 833,
    9: 908,
    10: 983,
    11: 1058,
}
PITCH: Final = 75
POSITION_OFFSETS: Final[tuple[int, ...]] = (-10, -5, 0, 5, 10)
PAIR_OFFSETS: Final[tuple[int, ...]] = (-15, -5, 5, 15)
SINGLE_THRESHOLDS: Final[tuple[int, ...]] = (100, 130, 160)
PAIR_THRESHOLDS: Final[tuple[int, ...]] = (120, 160)


def enhance(img_gray: np.ndarray, scale: int = 5) -> list[np.ndarray]:
    """Generate multiple enhanced variants of an image zone."""
    height, width = img_gray.shape

    def upscale(image: np.ndarray) -> np.ndarray:
        return cv2.resize(
            image,
            (width * scale, height * scale),
            interpolation=cv2.INTER_CUBIC,
        )

    variants: list[np.ndarray] = []
    for clip in (4, 8, 16, 32, 64):
        clahe = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(3, 3))
        applied = clahe.apply(img_gray)
        variants.extend([upscale(cv2.bitwise_not(applied)), upscale(applied)])

    kernel = np.ones((3, 3), np.uint8)
    variants.extend(
        [
            upscale(cv2.morphologyEx(img_gray, cv2.MORPH_GRADIENT, kernel)),
            upscale(cv2.bitwise_not(cv2.equalizeHist(img_gray))),
        ]
    )

    for gamma in (0.3, 0.5):
        lut = np.array([((index / 255.0) ** gamma) * 255 for index in range(256)], dtype=np.uint8)
        variants.append(upscale(cv2.bitwise_not(cv2.LUT(img_gray, lut))))

    for kernel_size in (7, 11):
        top_hat_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        variants.append(
            upscale(cv2.morphologyEx(img_gray, cv2.MORPH_TOPHAT, top_hat_kernel))
        )
    return variants


def _extract_digits(text: str) -> str:
    """Remove non-digit characters from OCR text."""
    return re.sub(r"\D", "", text)


def _iter_ocr_texts(
    image: np.ndarray, config: str, thresholds: tuple[int, ...]
) -> Iterable[str]:
    """Yield direct OCR output plus thresholded OCR outputs."""
    try:
        direct_text = pytesseract.image_to_string(image, config=config).strip()
    except OCR_EXCEPTIONS:
        direct_text = ""
    if direct_text:
        yield direct_text

    for threshold in thresholds:
        _, binary = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
        try:
            threshold_text = pytesseract.image_to_string(binary, config=config).strip()
        except OCR_EXCEPTIONS:
            continue
        if threshold_text:
            yield threshold_text


def _iter_single_digit_reads(image: np.ndarray) -> Iterable[str]:
    """Yield single-digit OCR hypotheses for one variant image."""
    for config in SINGLE_DIGIT_CONFIGS:
        for text in _iter_ocr_texts(image, config, SINGLE_THRESHOLDS):
            digits = _extract_digits(text)
            if digits:
                yield digits[0]


def _iter_pair_reads(image: np.ndarray) -> Iterable[str]:
    """Yield two-digit OCR hypotheses for one variant image."""
    for config in PAIR_CONFIGS:
        for text in _iter_ocr_texts(image, config, PAIR_THRESHOLDS):
            digits = _extract_digits(text)
            if len(digits) == 2:
                yield digits


def _position_window(center_x: float, img_width: int) -> tuple[int, int]:
    """Compute clipped search window around one x-center."""
    half_width = PITCH * 0.5
    x0 = max(0, int(center_x - half_width))
    x1 = min(img_width, int(center_x + half_width))
    return x0, x1


def scan_position(
    position: int, img_width: int, channels: dict[str, np.ndarray]
) -> Counter[str]:
    """Perform multi-channel, multi-offset scan for a single digit position."""
    base_center = CENTERS[position]
    votes: Counter[str] = Counter()

    for delta in POSITION_OFFSETS:
        x0, x1 = _position_window(base_center + delta, img_width)
        for channel_image in channels.values():
            zone = channel_image[:, x0:x1]
            if zone.size == 0:
                continue
            for variant in enhance(zone):
                for digit in _iter_single_digit_reads(variant):
                    votes[digit] += 1
    return votes


def _position_note(center_x: int) -> str:
    """Return visibility note for one digit center."""
    if 815 <= center_x <= 1031:
        return " [under marker]"
    if 700 < center_x < 815:
        return " [barely visible]"
    if 1032 <= center_x <= 1084:
        return " [edge visible]"
    return ""


def _print_position_votes(position: int, center_x: int, votes: Counter[str]) -> None:
    """Print one position voting summary."""
    total_votes = sum(votes.values())
    if not total_votes:
        print(f"  POS {position} (cx={center_x}): no reads")
        return

    ranked = votes.most_common(6)
    summary = ", ".join(
        f"'{digit}'={count / total_votes * 100:.0f}%"
        for digit, count in ranked[:5]
    )
    print(
        f"  POS {position} (cx={center_x}){_position_note(center_x)}: "
        f"{summary}  (n={total_votes})"
    )


def _pair_window(center_x: float, img_width: int) -> tuple[int, int]:
    """Compute clipped pair window centered between positions 10 and 11."""
    half_width = PITCH * 1.1
    x0 = max(0, int(center_x - half_width))
    x1 = min(img_width, int(center_x + half_width))
    return x0, x1


def scan_pair_votes(img_width: int, channels: dict[str, np.ndarray]) -> Counter[str]:
    """Collect pair votes for positions 10+11."""
    midpoint = (CENTERS[10] + CENTERS[11]) / 2
    pair_votes: Counter[str] = Counter()

    for delta in PAIR_OFFSETS:
        x0, x1 = _pair_window(midpoint + delta, img_width)
        for channel_image in channels.values():
            zone = channel_image[:, x0:x1]
            if zone.size == 0:
                continue
            for variant in enhance(zone):
                for pair in _iter_pair_reads(variant):
                    pair_votes[pair] += 1
    return pair_votes


def _print_pair_votes(votes: Counter[str]) -> None:
    """Print pair-vote histogram."""
    total_votes = sum(votes.values())
    if not total_votes:
        return
    for pair, count in votes.most_common(10):
        print(f"  '{pair}': {count:4d} ({count / total_votes * 100:5.1f}%)")


def _save_debug_images(gray_roi: np.ndarray, img_width: int) -> None:
    """Save debug crops for positions 11 and 10."""
    print("\n=== Debug images ===")
    half_width = int(PITCH * 0.6)
    pos11 = gray_roi[:, max(0, CENTERS[11] - half_width) : min(img_width, CENTERS[11] + half_width)]
    pos10 = gray_roi[:, max(0, CENTERS[10] - half_width) : min(img_width, CENTERS[10] + half_width)]
    cv2.imwrite("/tmp/pos11_correct_raw.png", pos11)
    cv2.imwrite("/tmp/pos10_correct_raw.png", pos10)

    clahe = cv2.createCLAHE(clipLimit=32.0, tileGridSize=(3, 3))
    for name, zone in (("pos11", pos11), ("pos10", pos10)):
        enhanced = cv2.bitwise_not(clahe.apply(zone))
        upscaled = cv2.resize(
            enhanced,
            (enhanced.shape[1] * 8, enhanced.shape[0] * 8),
            interpolation=cv2.INTER_CUBIC,
        )
        cv2.imwrite(f"/tmp/{name}_correct_enh.png", upscaled)
    print("  Debug images saved to /tmp/")


def _load_roi() -> tuple[np.ndarray, np.ndarray, int, int, int, int]:
    """Load source card image and compute OCR ROI."""
    image = cv2.imread(IMAGE_PATH)
    if image is None:
        raise FileNotFoundError(f"Could not read image at {IMAGE_PATH}")

    image_height, image_width = image.shape[:2]
    y0_roi, y1_roi = int(image_height * 0.25), int(image_height * 0.55)
    roi = image[y0_roi:y1_roi]
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    return roi, gray_roi, image_width, image_height, y0_roi, y1_roi


def _build_channels(roi: np.ndarray, gray_roi: np.ndarray) -> dict[str, np.ndarray]:
    """Build channel dictionary used by scanners."""
    return {
        "gray": gray_roi,
        "blue": roi[:, :, 0],
        "green": roi[:, :, 1],
        "red": roi[:, :, 2],
    }


def _scan_all_positions(img_width: int, channels: dict[str, np.ndarray]) -> None:
    """Run and print scans for positions 6..11."""
    print("\n=== Per-position scan (corrected coordinates) ===\n")
    for position in range(6, 12):
        center_x = CENTERS[position]
        votes = scan_position(position, img_width, channels)
        _print_position_votes(position, center_x, votes)


def main() -> None:
    """Main execution entry point."""
    roi, gray_roi, image_width, image_height, y0_roi, y1_roi = _load_roi()
    print(f"Image: {image_width}x{image_height}, ROI y=[{y0_roi},{y1_roi}]")

    channels = _build_channels(roi, gray_roi)
    _scan_all_positions(image_width, channels)

    print("\n=== PAIR: Positions 10+11 together ===")
    _print_pair_votes(scan_pair_votes(image_width, channels))
    _save_debug_images(gray_roi, image_width)


if __name__ == "__main__":
    main()
