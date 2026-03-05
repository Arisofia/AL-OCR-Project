"""Rescan all hidden PAN positions using multiple suffix-anchor offsets."""

from __future__ import annotations

import re
from collections import Counter
from typing import Final

import cv2
import numpy as np
import pytesseract

DEFAULT_IMAGE_PATH: Final = "/Users/jenineferderas/Desktop/card_image.jpg"
OX_CANDIDATES: Final[tuple[int, ...]] = (1030, 1050, 1070, 1090)
PITCH: Final[float] = 40.0
OCR_Y_RANGE: Final[tuple[int, int]] = (40, 160)
OCR_CONFIGS: Final[tuple[str, ...]] = (
    "--oem 3 --psm 10 -c tessedit_char_whitelist=0123456789",
    "--oem 3 --psm 13 -c tessedit_char_whitelist=0123456789",
)
THRESHOLDS: Final[tuple[int, ...]] = (120, 160)
OCR_EXCEPTIONS = (pytesseract.TesseractError, RuntimeError, TypeError, ValueError)
GAP_FACTORS: Final[tuple[float, ...]] = (0.0, 0.5, 1.0)


def load_roi(image_path: str = DEFAULT_IMAGE_PATH) -> tuple[np.ndarray, np.ndarray, int]:
    """Load source image and return ROI in BGR + grayscale with original width."""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")

    image_height, image_width = image.shape[:2]
    y_start, y_end = int(image_height * 0.20), int(image_height * 0.70)
    roi = image[y_start:y_end]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    return roi, gray, image_width


def enhance(gray: np.ndarray, scale: int = 5) -> list[np.ndarray]:
    """Build enhancement variants tuned for embossed card digits."""
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

    gamma = 0.3
    lut = np.array([((index / 255.0) ** gamma) * 255 for index in range(256)], dtype=np.uint8)
    variants.append(upscale(cv2.bitwise_not(cv2.LUT(gray, lut))))
    return variants


def extract_digits(text: str) -> str:
    """Remove non-digit characters from OCR output."""
    return re.sub(r"\D", "", text)


def ocr_single_digit(image: np.ndarray) -> list[str]:
    """Read one digit from an OCR variant with threshold fallbacks."""
    reads: list[str] = []
    for config in OCR_CONFIGS:
        try:
            digit_text = pytesseract.image_to_string(image, config=config).strip()
            digits = extract_digits(digit_text)
            if digits:
                reads.append(digits[0])
        except OCR_EXCEPTIONS:
            pass

        for threshold in THRESHOLDS:
            _, binary = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
            try:
                digit_text = pytesseract.image_to_string(binary, config=config).strip()
                digits = extract_digits(digit_text)
                if digits:
                    reads.append(digits[0])
            except OCR_EXCEPTIONS:
                pass
    return reads


def _iter_search_windows(position: int, width: int):
    """Yield horizontal search windows for one position."""
    offset_from_zero = 12 - position
    for x_zero in OX_CANDIDATES:
        for gap_factor in GAP_FACTORS:
            gap = PITCH * gap_factor
            center_x = x_zero - gap - PITCH * (offset_from_zero - 0.5)
            half_width = PITCH * 0.6
            x0 = max(0, int(center_x - half_width))
            x1 = min(width, int(center_x + half_width))
            if x1 - x0 >= 10:
                yield x0, x1


def _collect_zone_votes(
    votes: Counter[str], channels: dict[str, np.ndarray], x0: int, x1: int
) -> None:
    """Accumulate OCR votes for one horizontal zone across channels."""
    y0, y1 = OCR_Y_RANGE
    for channel_image in channels.values():
        zone = channel_image[y0:y1, x0:x1]
        if zone.size == 0:
            continue
        for variant in enhance(zone):
            for digit in ocr_single_digit(variant):
                votes[digit] += 1


def collect_votes(position: int, width: int, channels: dict[str, np.ndarray]) -> Counter[str]:
    """Collect OCR votes for one hidden card position."""
    votes: Counter[str] = Counter()
    for x0, x1 in _iter_search_windows(position, width):
        _collect_zone_votes(votes, channels, x0, x1)
    return votes


def print_votes(position: int, votes: Counter[str]) -> None:
    """Print normalized vote summary for one position."""
    total = sum(votes.values())
    if not total:
        print(f"  POS {position}: no reads")
        return

    ranked = votes.most_common(6)
    summary = ", ".join(f"'{digit}'={count / total * 100:.0f}%" for digit, count in ranked[:4])
    print(f"  POS {position}: {summary}  (n={total})")


def main() -> None:
    """Run position scan for all hidden digits."""
    roi, gray, image_width = load_roi()
    channels = {"gray": gray, "red": roi[:, :, 2]}

    for position in range(6, 12):
        votes = collect_votes(position, image_width, channels)
        print_votes(position, votes)


if __name__ == "__main__":
    main()
