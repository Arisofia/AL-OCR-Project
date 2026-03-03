#!/usr/bin/env python3
# pylint: disable=duplicate-code
"""Lean deep zoom on PAN positions 10-11 with minimal OCR calls."""

from __future__ import annotations

import re
import sys
from collections import Counter
from typing import TypedDict

import cv2
import numpy as np
import pytesseract

OCR_DIGIT_WHITELIST = "-c tessedit_char_whitelist=0123456789"
TESSERACT_PSM6 = "--oem 3 --psm 6"
TESSERACT_PSM7 = f"--oem 3 --psm 7 {OCR_DIGIT_WHITELIST}"
TESSERACT_PSM10 = f"--oem 3 --psm 10 {OCR_DIGIT_WHITELIST}"
TESSERACT_PSM11 = "--oem 3 --psm 11"
TESSERACT_PSM13 = f"--oem 3 --psm 13 {OCR_DIGIT_WHITELIST}"
SINGLE_DIGIT_CONFIGS = (TESSERACT_PSM10, TESSERACT_PSM13)
PAIR_CONFIGS = (TESSERACT_PSM7, TESSERACT_PSM13)
OCR_EXCEPTIONS = (pytesseract.TesseractError, RuntimeError, TypeError, ValueError)
DEFAULT_IMAGE_PATH = "/Users/jenineferderas/Desktop/card_image.jpg"


class CharBox(TypedDict):
    """Character-level box from OCR output."""

    ch: str
    x1: int
    y1: int
    x2: int
    y2: int


class DigitBox(TypedDict):
    """Digit box coordinates in ROI space."""

    x1: int
    y1: int
    x2: int
    y2: int


def load(path: str) -> np.ndarray:
    """Load image from path and abort if the file cannot be opened."""
    image = cv2.imread(path)
    if image is None:
        sys.exit(f"Cannot load {path}")
    return image


def char_boxes(img: np.ndarray, config: str = TESSERACT_PSM6) -> list[CharBox]:
    """Return char-level OCR boxes with top-left image coordinates."""
    data = pytesseract.image_to_boxes(img, config=config)
    height = img.shape[0]
    boxes: list[CharBox] = []
    for line in data.strip().split("\n"):
        parts = line.split()
        if len(parts) < 5:
            continue
        ch = parts[0]
        x1, y1, x2, y2 = (int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4]))
        boxes.append(
            {"ch": ch, "x1": x1, "y1": height - y2, "x2": x2, "y2": height - y1}
        )
    return boxes


def enhance(gray: np.ndarray, scale: int = 5) -> list[np.ndarray]:
    """Create a compact enhancement set tuned for embossed digits."""
    height, width = gray.shape

    def upscale(image: np.ndarray) -> np.ndarray:
        return cv2.resize(
            image,
            (width * scale, height * scale),
            interpolation=cv2.INTER_CUBIC,
        )

    enhanced: list[np.ndarray] = []
    for clip in (8, 32, 64):
        clahe = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(3, 3))
        applied = clahe.apply(gray)
        enhanced.append(upscale(cv2.bitwise_not(applied)))
        enhanced.append(upscale(applied))

    kernel = np.ones((3, 3), np.uint8)
    enhanced.append(upscale(cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)))
    enhanced.append(upscale(cv2.bitwise_not(cv2.equalizeHist(gray))))

    for gamma in (0.3, 0.5):
        lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8)
        enhanced.append(upscale(cv2.bitwise_not(cv2.LUT(gray, lut))))

    return enhanced


def extract_digits(text: str) -> str:
    """Strip non-digit characters from OCR output."""
    return re.sub(r"\D", "", text)


def ocr_single_digit(img: np.ndarray) -> list[str]:
    """Read one digit from an image using multiple OCR passes."""
    digits: list[str] = []

    for config in SINGLE_DIGIT_CONFIGS:
        try:
            read = extract_digits(pytesseract.image_to_string(img, config=config).strip())
            if read:
                digits.append(read[0])
        except OCR_EXCEPTIONS:
            pass

        for threshold in (120, 160):
            _, binary = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
            try:
                read = extract_digits(pytesseract.image_to_string(binary, config=config).strip())
                if read:
                    digits.append(read[0])
            except OCR_EXCEPTIONS:
                pass

    return digits


def ocr_pair(img: np.ndarray) -> list[str]:
    """Read a two-digit pair from an image using multiple OCR passes."""
    pairs: list[str] = []

    for config in PAIR_CONFIGS:
        try:
            read = extract_digits(pytesseract.image_to_string(img, config=config).strip())
            if len(read) == 2:
                pairs.append(read)
        except OCR_EXCEPTIONS:
            pass

        for threshold in (120, 160):
            _, binary = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
            try:
                read = extract_digits(pytesseract.image_to_string(binary, config=config).strip())
                if len(read) == 2:
                    pairs.append(read)
            except OCR_EXCEPTIONS:
                pass

    return pairs


def locate_suffix_boxes(gray: np.ndarray) -> list[DigitBox] | None:
    """Locate the '0665' suffix and return four digit boxes when found."""
    suffix_configs = (
        TESSERACT_PSM6,
        TESSERACT_PSM7.replace(OCR_DIGIT_WHITELIST, ""),
        TESSERACT_PSM11,
    )
    for config in suffix_configs:
        for clip in (8, 16, 32):
            clahe = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(4, 4))
            enhanced = clahe.apply(gray)
            for source in (cv2.bitwise_not(enhanced), enhanced):
                boxes = char_boxes(source, config)
                digit_entries: list[tuple[str, DigitBox]] = [
                    (
                        box["ch"],
                        {
                            "x1": box["x1"],
                            "x2": box["x2"],
                            "y1": box["y1"],
                            "y2": box["y2"],
                        },
                    )
                    for box in boxes
                    if box["ch"].isdigit()
                ]
                digit_text = "".join(character for character, _ in digit_entries)
                digit_boxes = [box for _, box in digit_entries]
                index = digit_text.rfind("0665")
                if index >= 0:
                    print(f"  Found! cfg={config}, clip={clip}")
                    print(f"  Digit text context: ...{digit_text[max(0, index - 6):index + 10]}...")
                    return digit_boxes[index:index + 4]
    return None


def fallback_suffix_boxes() -> list[DigitBox]:
    """Return estimated suffix boxes when OCR cannot locate '0665'."""
    print("  FALLBACK: Using estimated positions")
    return [
        {"x1": 1050, "x2": 1090, "y1": 40, "y2": 100},
        {"x1": 1090, "x2": 1130, "y1": 40, "y2": 100},
        {"x1": 1130, "x2": 1170, "y1": 40, "y2": 100},
        {"x1": 1170, "x2": 1210, "y1": 40, "y2": 100},
    ]


def print_ranked_votes(votes: Counter[str], limit: int = 8) -> None:
    """Print ranked vote distribution for OCR hypotheses."""
    total_votes = sum(votes.values())
    if not total_votes:
        print("      No reads")
        return

    for digit, count in votes.most_common(limit):
        percentage = count / total_votes * 100
        bar_graph = "#" * max(1, int(percentage / 2))
        print(f"      '{digit}': {count:4d} ({percentage:5.1f}%)  {bar_graph}")


def main() -> None:  # pylint: disable=too-many-locals,too-many-branches,too-many-statements  # NOSONAR
    """Run lean deep-zoom OCR on PAN positions 10 and 11."""
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IMAGE_PATH
    image = load(path)
    height, width = image.shape[:2]
    print(f"Image: {width}x{height}")

    y_start, y_end = int(height * 0.20), int(height * 0.70)
    roi = image[y_start:y_end]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    print("\n-- Locating '0665' --")
    found_0665 = locate_suffix_boxes(gray) or fallback_suffix_boxes()

    pitches = [found_0665[index + 1]["x1"] - found_0665[index]["x1"] for index in range(3)]
    pitch = sum(pitches) / len(pitches) if pitches else 40
    digit_height = found_0665[0]["y2"] - found_0665[0]["y1"]
    x_zero = found_0665[0]["x1"]
    y0 = max(0, found_0665[0]["y1"] - int(digit_height * 0.4))
    y1 = min(roi.shape[0], found_0665[0]["y2"] + int(digit_height * 0.4))

    print(f"  Digit pitch: {pitch:.1f}px, digit height: {digit_height}px")
    print(f"  '0' of 0665 at x={x_zero}")

    print("\n-- Deep scan positions 10 and 11 --\n")
    for position, label in ((11, "POS 11 (1 left of 0665)"), (10, "POS 10 (2 left of 0665)")):
        print(f"  {label}")
        votes: Counter[str] = Counter()
        offset_digits = 12 - position

        for gap_factor in (0.0, 0.3, 0.5, 0.7, 1.0, 1.3):
            gap = pitch * gap_factor
            center_x = x_zero - gap - pitch * (offset_digits - 0.5)
            half_width = pitch * 0.6
            x0 = max(0, int(center_x - half_width))
            x1 = min(width, int(center_x + half_width))

            zone = gray[y0:y1, x0:x1]
            if zone.size > 0:
                for variant in enhance(zone, scale=5):
                    for read in ocr_single_digit(variant):
                        votes[read] += 1

            for channel in range(3):
                zone_channel = roi[y0:y1, x0:x1, channel]
                if zone_channel.size == 0:
                    continue
                for variant in enhance(zone_channel, scale=5):
                    for read in ocr_single_digit(variant):
                        votes[read] += 1

        print_ranked_votes(votes)
        print()

    print("  PAIR (pos 10+11 together)")
    pair_votes: Counter[str] = Counter()
    for gap_factor in (0.0, 0.3, 0.5, 0.7, 1.0):
        gap = pitch * gap_factor
        center_x = x_zero - gap - pitch
        half_width = pitch * 1.2
        x0 = max(0, int(center_x - half_width))
        x1 = min(width, int(center_x + half_width))
        zone = gray[y0:y1, x0:x1]
        if zone.size == 0:
            continue
        for variant in enhance(zone, scale=5):
            for pair in ocr_pair(variant):
                pair_votes[pair] += 1

    total_pair_votes = sum(pair_votes.values())
    if total_pair_votes:
        for pair, count in pair_votes.most_common(10):
            percentage = count / total_pair_votes * 100
            print(f"      '{pair}': {count:4d} ({percentage:5.1f}%)")
    print()

    print("  TRANSITION (last 3 hidden -> 0665)")
    transition_votes: Counter[str] = Counter()
    for shift in range(-20, 21, 10):
        x0 = max(0, int(x_zero - pitch * 3.5 + shift))
        x1 = min(width, found_0665[3]["x2"] + 5)
        zone = gray[y0:y1, x0:x1]
        if zone.size == 0:
            continue
        for variant in enhance(zone, scale=4):
            for config in PAIR_CONFIGS:
                try:
                    text = pytesseract.image_to_string(variant, config=config).strip()
                    read = extract_digits(text)
                    if len(read) >= 4:
                        transition_votes[read] += 1
                except OCR_EXCEPTIONS:
                    pass

    total_transition_votes = sum(transition_votes.values())
    if total_transition_votes:
        for sequence, count in transition_votes.most_common(15):
            percentage = count / total_transition_votes * 100
            if sequence.endswith("0665"):
                tag = " <<"
            elif sequence.endswith("665"):
                tag = " <"
            else:
                tag = ""
            print(f"      '{sequence}': {count:4d} ({percentage:5.1f}%){tag}")

    print("\n-- Debug images --")
    for gap_factor in (0.0, 0.5):
        gap = pitch * gap_factor
        for position, label in ((11, "p11"), (10, "p10")):
            offset = 12 - position
            center_x = x_zero - gap - pitch * (offset - 0.5)
            half_width = pitch * 0.6
            x0 = max(0, int(center_x - half_width))
            x1 = min(width, int(center_x + half_width))
            zone = gray[y0:y1, x0:x1]
            cv2.imwrite(f"/tmp/dz2_{label}_g{gap_factor}.png", zone)

            clahe = cv2.createCLAHE(clipLimit=32.0, tileGridSize=(3, 3))
            enhanced = cv2.bitwise_not(clahe.apply(zone))
            big = cv2.resize(
                enhanced,
                (enhanced.shape[1] * 8, enhanced.shape[0] * 8),
                interpolation=cv2.INTER_CUBIC,
            )
            cv2.imwrite(f"/tmp/dz2_{label}_g{gap_factor}_enh.png", big)
            print(f"  /tmp/dz2_{label}_g{gap_factor}.png & _enh.png ({x0}-{x1})")


if __name__ == "__main__":
    main()
