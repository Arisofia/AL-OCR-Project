#!/usr/bin/env python3
"""Quick scan of PAN positions 10 and 11 using fallback coordinates."""

from __future__ import annotations

import re
from collections import Counter

import cv2
import numpy as np
import pytesseract

DEFAULT_IMAGE_PATH = "/Users/jenineferderas/Desktop/card_image.jpg"
OCR_DIGIT_WHITELIST = "-c tessedit_char_whitelist=0123456789"
TESSERACT_PSM7 = f"--oem 3 --psm 7 {OCR_DIGIT_WHITELIST}"
TESSERACT_PSM10 = f"--oem 3 --psm 10 {OCR_DIGIT_WHITELIST}"
TESSERACT_PSM13 = f"--oem 3 --psm 13 {OCR_DIGIT_WHITELIST}"
SINGLE_DIGIT_CONFIGS = (TESSERACT_PSM10, TESSERACT_PSM13)
PAIR_CONFIGS = (TESSERACT_PSM7, TESSERACT_PSM13)
THRESHOLDS = (120, 160)
OCR_EXCEPTIONS = (pytesseract.TesseractError, RuntimeError, TypeError, ValueError)

# Fallback geometry based on prior deep-zoom runs.
FALLBACK_PITCH = 40.0
FALLBACK_X_ZERO = 1050.0
FALLBACK_Y_RANGE = (40, 160)


def load_roi(image_path: str = DEFAULT_IMAGE_PATH) -> tuple[np.ndarray, np.ndarray]:
    """Load image and return PAN ROI in BGR and grayscale forms."""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")

    height, _ = image.shape[:2]
    y_start, y_end = int(height * 0.20), int(height * 0.70)
    roi = image[y_start:y_end]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    return roi, gray


def enhance(gray: np.ndarray, scale: int = 5) -> list[np.ndarray]:
    """Create a compact enhancement set for embossed-digit OCR."""
    height, width = gray.shape

    def upscale(image: np.ndarray) -> np.ndarray:
        return cv2.resize(
            image,
            (width * scale, height * scale),
            interpolation=cv2.INTER_CUBIC,
        )

    variants: list[np.ndarray] = []
    for clip in (8, 32, 64):
        clahe = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(3, 3))
        applied = clahe.apply(gray)
        variants.append(upscale(cv2.bitwise_not(applied)))
        variants.append(upscale(applied))

    kernel = np.ones((3, 3), np.uint8)
    variants.append(upscale(cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)))
    variants.append(upscale(cv2.bitwise_not(cv2.equalizeHist(gray))))

    for gamma in (0.3, 0.5):
        lut = np.array([((idx / 255.0) ** gamma) * 255 for idx in range(256)], dtype=np.uint8)
        variants.append(upscale(cv2.bitwise_not(cv2.LUT(gray, lut))))

    return variants


def extract_digits(text: str) -> str:
    """Strip non-digit characters from OCR output."""
    return re.sub(r"\D", "", text)


def ocr_single_digit(image: np.ndarray) -> list[str]:
    """Read one digit using multiple OCR configs and thresholds."""
    reads: list[str] = []

    for config in SINGLE_DIGIT_CONFIGS:
        try:
            digit_str = extract_digits(pytesseract.image_to_string(image, config=config).strip())
            if digit_str:
                reads.append(digit_str[0])
        except OCR_EXCEPTIONS:
            pass

        for threshold in THRESHOLDS:
            _, binary = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
            try:
                digit_text = pytesseract.image_to_string(binary, config=config).strip()
                digit_str = extract_digits(digit_text)
                if digit_str:
                    reads.append(digit_str[0])
            except OCR_EXCEPTIONS:
                pass

    return reads


def ocr_pair(image: np.ndarray) -> list[str]:
    """Read two-digit combinations using multiple OCR configs and thresholds."""
    reads: list[str] = []

    for config in PAIR_CONFIGS:
        try:
            digits = extract_digits(pytesseract.image_to_string(image, config=config).strip())
            if len(digits) == 2:
                reads.append(digits)
        except OCR_EXCEPTIONS:
            pass

        for threshold in THRESHOLDS:
            _, binary = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
            try:
                digits = extract_digits(pytesseract.image_to_string(binary, config=config).strip())
                if len(digits) == 2:
                    reads.append(digits)
            except OCR_EXCEPTIONS:
                pass

    return reads


def build_channels(roi: np.ndarray, gray: np.ndarray) -> tuple[np.ndarray, ...]:
    """Return grayscale and BGR channels used during OCR voting."""
    return (gray, roi[:, :, 0], roi[:, :, 1], roi[:, :, 2])


# pylint: disable=too-many-locals
def scan_position(
    channels: tuple[np.ndarray, ...],
    width: int,
    center_multiplier: float,
) -> Counter[str]:
    """Collect OCR votes for one position defined by center multiplier."""
    votes: Counter[str] = Counter()
    y0, y1 = FALLBACK_Y_RANGE

    for gap_factor in (0.0, 0.3, 0.5, 0.7, 1.0, 1.3):
        gap = FALLBACK_PITCH * gap_factor
        center_x = FALLBACK_X_ZERO - gap - FALLBACK_PITCH * center_multiplier
        half_width = FALLBACK_PITCH * 0.6
        x0 = max(0, int(center_x - half_width))
        x1 = min(width, int(center_x + half_width))

        for channel_image in channels:
            zone = channel_image[y0:y1, x0:x1]
            if zone.size == 0:
                continue
            for variant in enhance(zone):
                for digit in ocr_single_digit(variant):
                    votes[digit] += 1

    return votes
# pylint: enable=too-many-locals


def scan_pair(gray: np.ndarray, width: int) -> Counter[str]:
    """Collect OCR votes for the two-digit pair (positions 10+11)."""
    votes: Counter[str] = Counter()
    y0, y1 = FALLBACK_Y_RANGE

    for gap_factor in (0.0, 0.3, 0.5, 0.7, 1.0):
        gap = FALLBACK_PITCH * gap_factor
        center_x = FALLBACK_X_ZERO - gap - FALLBACK_PITCH
        half_width = FALLBACK_PITCH * 1.2
        x0 = max(0, int(center_x - half_width))
        x1 = min(width, int(center_x + half_width))

        zone = gray[y0:y1, x0:x1]
        if zone.size == 0:
            continue
        for variant in enhance(zone):
            for pair in ocr_pair(variant):
                votes[pair] += 1

    return votes


def print_ranked_votes(title: str, votes: Counter[str], limit: int = 8) -> None:
    """Print ranked single-digit OCR votes."""
    print(title)
    total = sum(votes.values())
    if not total:
        print("  No reads")
        return

    for digit, count in votes.most_common(limit):
        percentage = count / total * 100
        bar_graph = "#" * max(1, int(percentage / 2))
        print(f"  '{digit}': {count:4d} ({percentage:5.1f}%)  {bar_graph}")


def print_pair_votes(votes: Counter[str], limit: int = 10) -> None:
    """Print ranked pair OCR votes."""
    print("\nPAIR (pos 10+11)")
    total = sum(votes.values())
    if not total:
        print("  No reads")
        return

    for pair, count in votes.most_common(limit):
        percentage = count / total * 100
        print(f"  '{pair}': {count:4d} ({percentage:5.1f}%)")


def main() -> None:
    """Run quick fallback OCR scan for positions 10 and 11."""
    roi, gray = load_roi()
    channels = build_channels(roi, gray)
    width = gray.shape[1]

    pos10_votes = scan_position(channels, width, center_multiplier=1.5)
    print_ranked_votes("POS 10 (2 digits left of 0665)", pos10_votes)

    pos11_votes = scan_position(channels, width, center_multiplier=0.5)
    print_ranked_votes("\nPOS 11 (1 digit left of 0665)", pos11_votes)

    pair_votes = scan_pair(gray, width)
    print_pair_votes(pair_votes)


if __name__ == "__main__":
    main()
