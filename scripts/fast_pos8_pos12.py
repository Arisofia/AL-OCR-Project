#!/usr/bin/env python3
"""Fast re-examination of PAN positions 8 and 12."""

from __future__ import annotations

import re
import sys
from collections import Counter
from dataclasses import dataclass
from itertools import product
from typing import Final

import cv2
import numpy as np
import pytesseract

IMG_PATH = (
    sys.argv[1]
    if len(sys.argv) > 1
    else "/Users/jenineferderas/Desktop/card_image.jpg"
)
OCR_WHITELIST: Final = "-c tessedit_char_whitelist=0123456789"
OCR_CONFIGS: Final[tuple[str, ...]] = (
    f"--oem 3 --psm 10 {OCR_WHITELIST}",
    f"--oem 3 --psm 13 {OCR_WHITELIST}",
)
THRESHOLDS: Final[tuple[int, ...]] = (0, 120, 160)
OCR_EXCEPTIONS = (pytesseract.TesseractError, RuntimeError, TypeError, ValueError)
GAP_FACTORS: Final[tuple[float, ...]] = (0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0)
OFFSET_POS12: Final[tuple[int, ...]] = (-8, -4, 0, 4, 8)
OFFSET_POS8: Final[tuple[int, ...]] = (-6, 0, 6)
FLOAT_EPSILON: Final = 1e-9


@dataclass(frozen=True)
class DigitBox:
    """Digit-level OCR box in row coordinates."""

    ch: str
    x1: int
    y1: int
    x2: int
    y2: int


@dataclass(frozen=True)
class SuffixDetection:
    """Best suffix detection anchored on the '665' sequence."""

    boxes: tuple[DigitBox, ...]
    text: str
    idx_665: int
    label: str


@dataclass(frozen=True)
class ScanContext:
    """Geometry derived from suffix detection or fallback."""

    pitch: float
    y0: int
    y1: int
    x12_center: float


def _safe_first_digit(source: np.ndarray, config: str) -> str | None:
    """Run OCR on a source image and return first digit when available."""
    try:
        text = pytesseract.image_to_string(source, config=config).strip()
    except OCR_EXCEPTIONS:
        return None
    digits = re.sub(r"\D", "", text)
    return digits[0] if digits else None


def _upscale(image: np.ndarray, width: int, height: int, scale: int) -> np.ndarray:
    """Upscale image for OCR."""
    return cv2.resize(image, (width * scale, height * scale), interpolation=cv2.INTER_CUBIC)


def _build_variants(gray_zone: np.ndarray, scale: int) -> list[np.ndarray]:
    """Build the lean enhancement set used by fast OCR."""
    height, width = gray_zone.shape[:2]
    variants: list[np.ndarray] = []

    for clip in (8, 32, 64):
        clahe = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(3, 3))
        enhanced = clahe.apply(gray_zone)
        variants.append(_upscale(cv2.bitwise_not(enhanced), width, height, scale))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    top_hat = cv2.morphologyEx(gray_zone, cv2.MORPH_TOPHAT, kernel)
    variants.append(_upscale(top_hat, width, height, scale))

    lut = np.array([((idx / 255.0) ** 0.3) * 255 for idx in range(256)], dtype=np.uint8)
    gamma = cv2.LUT(gray_zone, lut)
    variants.append(_upscale(cv2.bitwise_not(gamma), width, height, scale))
    return variants


def _apply_threshold(image: np.ndarray, threshold: int) -> np.ndarray:
    """Return thresholded image or original when threshold is zero."""
    if threshold == 0:
        return image
    return cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)[1]


def ocr_zone_fast(gray_zone: np.ndarray, scale: int = 8) -> Counter[str]:
    """Fast OCR for one digit zone using lean variants and configs."""
    votes: Counter[str] = Counter()
    height, width = gray_zone.shape[:2]
    if height == 0 or width == 0:
        return votes

    variants = _build_variants(gray_zone, scale)
    for variant, config, threshold in product(variants, OCR_CONFIGS, THRESHOLDS):
        source = _apply_threshold(variant, threshold)
        if digit := _safe_first_digit(source, config):
            votes[digit] += 1
    return votes


def _parse_digit_boxes(box_data: str, row_height: int) -> list[DigitBox]:
    """Parse Tesseract char boxes and keep only digits."""
    boxes: list[DigitBox] = []
    for line in box_data.strip().split("\n"):
        parts = line.split()
        if len(parts) < 5 or not parts[0].isdigit():
            continue
        boxes.append(
            DigitBox(
                ch=parts[0],
                x1=int(parts[1]),
                y1=row_height - int(parts[4]),
                x2=int(parts[3]),
                y2=row_height - int(parts[2]),
            )
        )
    return boxes


def _best_suffix_detection(gray_row: np.ndarray) -> SuffixDetection | None:
    """Find the best detection containing the '665' sequence."""
    row_height = gray_row.shape[0]
    clahe = cv2.createCLAHE(clipLimit=16.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(gray_row)

    best: SuffixDetection | None = None
    for psm in (6, 7, 11):
        for source, label in (
            (cv2.bitwise_not(enhanced), "inv"),
            (enhanced, "raw"),
        ):
            try:
                box_data = pytesseract.image_to_boxes(
                    source, config=f"--oem 3 --psm {psm}"
                )
            except OCR_EXCEPTIONS:
                continue

            digit_boxes = _parse_digit_boxes(box_data, row_height)
            text = "".join(box.ch for box in digit_boxes)
            idx_665 = text.rfind("665")
            if idx_665 < 0:
                continue

            candidate = SuffixDetection(
                boxes=tuple(digit_boxes),
                text=text,
                idx_665=idx_665,
                label=f"{label}/psm{psm}",
            )
            if best is None or len(candidate.text) > len(best.text):
                best = candidate
    return best


def _context_from_detection(detection: SuffixDetection, row_height: int) -> ScanContext:
    """Build scanning geometry from suffix detection."""
    first_six = detection.boxes[detection.idx_665]
    second_six = detection.boxes[detection.idx_665 + 1]
    five = detection.boxes[detection.idx_665 + 2]
    pitch = ((second_six.x1 - first_six.x1) + (five.x1 - second_six.x1)) / 2

    digit_height = max(first_six.y2 - first_six.y1, five.y2 - five.y1)
    margin = int(digit_height * 0.3)
    y0 = max(0, min(first_six.y1, five.y1) - margin)
    y1 = min(row_height, max(first_six.y2, five.y2) + margin)

    if detection.idx_665 > 0:
        previous = detection.boxes[detection.idx_665 - 1]
        x12_center = (previous.x1 + previous.x2) / 2
    else:
        x12_center = first_six.x1 - pitch
    return ScanContext(pitch=pitch, y0=y0, y1=y1, x12_center=x12_center)


def _fallback_context(row_width: int, row_height: int) -> ScanContext:
    """Fallback geometry when no suffix is detected by char boxes."""
    pitch = 75 if row_width < 1200 else 94
    x12_center = row_width - 4 * pitch - 20
    return ScanContext(
        pitch=float(pitch),
        y0=int(row_height * 0.15),
        y1=int(row_height * 0.85),
        x12_center=float(x12_center),
    )


def _zone_window(center_x: float, half_width: float, row_width: int) -> tuple[int, int]:
    """Compute clipped zone window around center."""
    x0 = max(0, int(center_x - half_width))
    x1 = min(row_width, int(center_x + half_width))
    return x0, x1


def _scan_center(gray_row: np.ndarray, y0: int, y1: int, x0: int, x1: int) -> Counter[str]:
    """OCR one grayscale zone."""
    return ocr_zone_fast(gray_row[y0:y1, x0:x1])


def _scan_center_rgb(
    row_bgr: np.ndarray, y0: int, y1: int, x0: int, x1: int
) -> Counter[str]:
    """OCR one zone across all RGB channels."""
    votes: Counter[str] = Counter()
    for channel in range(3):
        votes += ocr_zone_fast(row_bgr[y0:y1, x0:x1, channel])
    return votes


def _scan_pos12_votes(gray_row: np.ndarray, row_bgr: np.ndarray, ctx: ScanContext) -> Counter[str]:
    """Collect votes for position 12."""
    votes: Counter[str] = Counter()
    half_width = ctx.pitch * 0.6
    row_width = gray_row.shape[1]

    for dx in OFFSET_POS12:
        x0, x1 = _zone_window(ctx.x12_center + dx, half_width, row_width)
        votes += _scan_center(gray_row, ctx.y0, ctx.y1, x0, x1)
        if dx == 0:
            votes += _scan_center_rgb(row_bgr, ctx.y0, ctx.y1, x0, x1)
    return votes


def _scan_pos8_votes(gray_row: np.ndarray, row_bgr: np.ndarray, ctx: ScanContext) -> Counter[str]:
    """Collect votes for position 8 using group-gap hypotheses."""
    votes: Counter[str] = Counter()
    half_width = ctx.pitch * 0.6
    row_width = gray_row.shape[1]

    for gap_factor in GAP_FACTORS:
        gap = ctx.pitch * gap_factor
        x8_center = ctx.x12_center - gap - 3.5 * ctx.pitch
        for dx in OFFSET_POS8:
            x0, x1 = _zone_window(x8_center + dx, half_width, row_width)
            votes += _scan_center(gray_row, ctx.y0, ctx.y1, x0, x1)

        if abs(gap_factor - 0.4) < FLOAT_EPSILON:
            x0, x1 = _zone_window(x8_center, half_width, row_width)
            votes += _scan_center_rgb(row_bgr, ctx.y0, ctx.y1, x0, x1)
    return votes


def _save_debug_images(
    gray_row: np.ndarray, ctx: ScanContext, center_x: float, prefix: str
) -> None:
    """Save raw and enhanced debug images for one scanned position."""
    half_width = ctx.pitch * 0.6
    row_width = gray_row.shape[1]
    x0, x1 = _zone_window(center_x, half_width, row_width)
    zone = gray_row[ctx.y0 : ctx.y1, x0:x1]
    cv2.imwrite(f"/tmp/{prefix}_raw.png", zone)

    clahe = cv2.createCLAHE(clipLimit=32.0, tileGridSize=(3, 3))
    enhanced = cv2.bitwise_not(clahe.apply(zone))
    zoom = cv2.resize(
        enhanced,
        (zone.shape[1] * 10, zone.shape[0] * 10),
        interpolation=cv2.INTER_CUBIC,
    )
    cv2.imwrite(f"/tmp/{prefix}_enh10x.png", zoom)


def _print_votes(label: str, votes: Counter[str], limit: int = 8) -> None:
    """Print histogram of OCR votes."""
    total_reads = sum(votes.values())
    print(f"\n{label} — {total_reads} total reads:")
    for digit, count in votes.most_common(limit):
        percentage = count / total_reads * 100 if total_reads else 0
        histogram = "#" * max(1, int(percentage / 2))
        print(f"  '{digit}': {count:4d} ({percentage:5.1f}%)  {histogram}")


def main() -> None:
    """Run fast focused scan for PAN positions 8 and 12."""
    image = cv2.imread(IMG_PATH)
    if image is None:
        sys.exit(f"Cannot load {IMG_PATH}")

    img_height, img_width = image.shape[:2]
    print(f"Image: {img_width}x{img_height}")
    y0, y1 = int(img_height * 0.30), int(img_height * 0.62)
    row = image[y0:y1]
    gray = cv2.cvtColor(row, cv2.COLOR_BGR2GRAY)
    row_height, row_width = gray.shape

    detection = _best_suffix_detection(gray)
    if detection is None:
        print("\nWARNING: '665' not found via char boxes. Using fallback geometry.")
        context = _fallback_context(row_width, row_height)
    else:
        print(f"\nFound '665' via {detection.label}: digits='{detection.text}'")
        context = _context_from_detection(detection, row_height)
        print(f"POS 12 center: x={context.x12_center:.0f}")

    print(
        f"Pitch: {context.pitch:.1f}px, "
        f"y=[{context.y0},{context.y1}]"
    )

    print(f"\n{'=' * 50}")
    print("POSITION 12 (first digit of last group)")
    print(f"{'=' * 50}")
    pos12_votes = _scan_pos12_votes(gray, row, context)
    _print_votes("POS 12", pos12_votes)
    _save_debug_images(gray, context, context.x12_center, "pos12")

    print(f"\n{'=' * 50}")
    print("POSITION 8 (3rd hidden digit)")
    print(f"{'=' * 50}")
    pos8_votes = _scan_pos8_votes(gray, row, context)
    _print_votes("POS 8", pos8_votes)
    x8_debug = context.x12_center - 0.4 * context.pitch - 3.5 * context.pitch
    _save_debug_images(gray, context, x8_debug, "pos8")

    best_12 = pos12_votes.most_common(1)[0][0] if pos12_votes else "?"
    best_8 = pos8_votes.most_common(1)[0][0] if pos8_votes else "?"

    total_12 = sum(pos12_votes.values())
    total_8 = sum(pos8_votes.values())
    top5_12 = (
        ", ".join(f"'{digit}'={count / total_12:.0%}" for digit, count in pos12_votes.most_common(5))
        if total_12
        else "?"
    )
    top5_8 = (
        ", ".join(f"'{digit}'={count / total_8:.0%}" for digit, count in pos8_votes.most_common(5))
        if total_8
        else "?"
    )

    print(f"\n{'=' * 50}")
    print("SUMMARY")
    print(f"{'=' * 50}")
    print(f"  POS 12: {top5_12}")
    print(f"  POS  8: {top5_8}")
    print(f"\n  Updated PAN: 4388 54?{best_8} ???? {best_12}665")
    print("  Debug images: /tmp/pos8_*.png, /tmp/pos12_*.png")


if __name__ == "__main__":
    main()
