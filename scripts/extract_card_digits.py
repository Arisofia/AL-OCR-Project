#!/usr/bin/env python3
"""
Targeted card digit extraction from embossed card images.

Specifically designed for:
- Silver/grey embossed text on dark card background
- Partial occlusion by marker/redaction (black, green, etc.)
- Leverages repo's PixelReconstructor + Tesseract digit modes

Usage:
  python3 scripts/extract_card_digits.py /Users/jenineferderas/Desktop/card_image.jpg
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytesseract

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ocr_reconstruct.modules.reconstruct import (  # pylint: disable=wrong-import-position,import-error
    PixelReconstructor,
)

try:
    from ocr_service.modules.personal_doc_extractor import (  # pylint: disable=wrong-import-position,import-error
        _luhn_valid as luhn_valid,
    )
except ImportError:

    def luhn_valid(number: str) -> bool:
        """Fallback Luhn validation when service module is unavailable."""
        if not number.isdigit() or not 13 <= len(number) <= 19:
            return False
        total = 0
        for index, char in enumerate(reversed(number)):
            digit = int(char)
            if index % 2 == 1:
                digit = digit * 2 - 9 if digit > 4 else digit * 2
            total += digit
        return total % 10 == 0


def load_and_inspect(path: str) -> np.ndarray:
    """Load image and print basic info."""
    img = cv2.imread(path)
    if img is None:
        print(f"ERROR: Cannot load {path}")
        sys.exit(1)

    height, width = img.shape[:2]
    channels = img.shape[2] if len(img.shape) == 3 else 1
    print(f"Loaded: {path}")
    print(f"  Size: {width}x{height}, channels: {channels}")
    return img


def extract_card_number_region(img: np.ndarray) -> np.ndarray:
    """
    Crop to the approximate card number region.
    Card numbers typically occupy the middle horizontal band,
    roughly 30%-60% vertically and 5%-95% horizontally.
    """
    height, width = img.shape[:2]
    y_start = int(height * 0.25)
    y_end = int(height * 0.65)
    x_start = int(width * 0.02)
    x_end = int(width * 0.98)
    roi = img[y_start:y_end, x_start:x_end]
    print(
        "  Card number ROI: "
        f"({x_start},{y_start})-({x_end},{y_end}) = "
        f"{roi.shape[1]}x{roi.shape[0]}"
    )
    return roi


def remove_dark_occlusion(img: np.ndarray) -> np.ndarray:
    """
    Detect dark marker/redaction areas and inpaint them.
    For black markers on dark cards, use adaptive thresholding
    to find the very darkest regions (marker) vs card background.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    _, dark_mask = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    marker_mask = np.zeros_like(dark_mask)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        _x, _y, width, height = cv2.boundingRect(cnt)
        if area > 500 and width > 30 and height > 20:
            contour = np.asarray(cnt, dtype=np.int32)
            cv2.drawContours(marker_mask, [contour], -1, (255, 255, 255), -1)

    kernel = np.ones((5, 5), np.uint8)
    marker_mask = cv2.dilate(marker_mask, kernel, iterations=2)

    ratio = float(np.count_nonzero(marker_mask)) / float(marker_mask.size or 1)
    print(f"  Dark occlusion detected: {ratio:.1%} of image")

    if ratio < 0.01:
        print("  No significant dark occlusion found")
        return img

    telea = cv2.inpaint(img, marker_mask, 7, cv2.INPAINT_TELEA)
    return cv2.inpaint(telea, marker_mask, 3, cv2.INPAINT_NS)


def enhance_embossed_text(img: np.ndarray) -> np.ndarray:
    """
    Enhance embossed silver/grey text on dark background.
    Uses CLAHE + inversion to make text dark-on-white for OCR.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    inverted = cv2.bitwise_not(enhanced)
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    return cv2.filter2D(inverted, -1, kernel)


def aggressive_digit_extraction(img: np.ndarray) -> list[dict[str, Any]]:
    """
    Run Tesseract in multiple modes optimized for digit extraction.
    Returns list of OCR attempts with extracted digit strings.
    """
    results: list[dict[str, Any]] = []

    configs: list[tuple[str, str]] = [
        ("PSM6-digits", "--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789"),
        ("PSM7-digits", "--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789"),
        ("PSM6-full", "--oem 3 --psm 6"),
        ("PSM11-digits", "--oem 3 --psm 11 -c tessedit_char_whitelist=0123456789"),
        ("PSM13-single", "--oem 3 --psm 13 -c tessedit_char_whitelist=0123456789"),
        ("PSM4-columns", "--oem 3 --psm 4"),
        (
            "PSM6-digits-nolstm",
            "--oem 0 --psm 6 -c tessedit_char_whitelist=0123456789",
        ),
    ]

    for mode_name, config in configs:
        try:
            text = pytesseract.image_to_string(img, config=config).strip()
            digits = re.sub(r"\D", "", text)
            if digits or text:
                results.append(
                    {
                        "mode": mode_name,
                        "text": text,
                        "digits": digits,
                        "digit_count": len(digits),
                    }
                )
        except (pytesseract.TesseractError, RuntimeError, TypeError, ValueError) as exc:
            results.append({"mode": mode_name, "error": str(exc)})

    return results


def try_multiple_thresholds(gray: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """Generate multiple threshold variants for OCR attempts."""
    variants: list[tuple[str, np.ndarray]] = []

    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(("otsu", otsu))

    adapt_mean = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        11,
        2,
    )
    variants.append(("adaptive-mean", adapt_mean))

    adapt_gauss = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2,
    )
    variants.append(("adaptive-gauss", adapt_gauss))

    for thresh_val in [80, 100, 120, 140, 160, 180]:
        _, fixed = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
        variants.append((f"fixed-{thresh_val}", fixed))

    return variants


def upscale(img: np.ndarray, factor: int = 3) -> np.ndarray:
    """Upscale image for better OCR resolution."""
    height, width = img.shape[:2]
    return cv2.resize(img, (width * factor, height * factor), interpolation=cv2.INTER_CUBIC)


def collect_valid_results(
    source_name: str,
    threshold_name: str,
    variant_img: np.ndarray,
    all_results: list[dict[str, Any]],
) -> None:
    """Run OCR over a threshold variant and append usable digit results."""
    results = aggressive_digit_extraction(variant_img)
    for result in results:
        digits = result.get("digits", "")
        if "digits" in result and len(digits) >= 4:
            result["source"] = source_name
            result["threshold"] = threshold_name
            all_results.append(result)


def process_source(
    source_name: str,
    source_img: np.ndarray,
    all_results: list[dict[str, Any]],
) -> None:
    """Process one source image through ROI->enhance->threshold OCR pipeline."""
    print(f"\n--- Processing: {source_name} ---")
    roi = extract_card_number_region(source_img)

    print("  Step 3: Enhance embossed text")
    enhanced = enhance_embossed_text(roi)
    cv2.imwrite(f"/tmp/card_{source_name}_enhanced.png", enhanced)

    enhanced_big = upscale(enhanced, 3)
    cv2.imwrite(f"/tmp/card_{source_name}_enhanced_3x.png", enhanced_big)

    print("  Step 4: Try multiple threshold variants")
    for variant_name, variant_img in try_multiple_thresholds(enhanced_big):
        collect_valid_results(source_name, variant_name, variant_img, all_results)


def process_full_image(img: np.ndarray, all_results: list[dict[str, Any]]) -> None:
    """Run additional OCR pass on full image variants."""
    print("\n--- Processing: full-image direct ---")
    gray_full = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inverted_full = cv2.bitwise_not(gray_full)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    enhanced_full = clahe.apply(inverted_full)
    big_full = upscale(enhanced_full, 2)

    for variant_name, variant_img in try_multiple_thresholds(big_full):
        collect_valid_results("full-image", variant_name, variant_img, all_results)


def process_depixelated(
    reconstructor: PixelReconstructor,
    cleaned_dark: np.ndarray,
    all_results: list[dict[str, Any]],
) -> None:
    """Run OCR pass after depixelation/reconstruction."""
    print("\n--- Processing: depixelated ---")
    depix = reconstructor.handle_pixelation(cleaned_dark)
    roi_depix = extract_card_number_region(depix)
    enhanced_depix = enhance_embossed_text(roi_depix)
    big_depix = upscale(enhanced_depix, 3)

    for variant_name, variant_img in try_multiple_thresholds(big_depix):
        collect_valid_results("depixelated", variant_name, variant_img, all_results)


def dedupe_results(all_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate OCR results by digit sequence and keep best ordering."""
    seen: set[str] = set()
    unique_results: list[dict[str, Any]] = []
    ranked = sorted(all_results, key=lambda row: -int(row.get("digit_count", 0)))

    for row in ranked:
        digits = str(row.get("digits", ""))
        if digits and digits not in seen:
            seen.add(digits)
            unique_results.append(row)
    return unique_results


def print_results(unique_results: list[dict[str, Any]]) -> None:
    """Print OCR candidates, highlights, and Luhn/pattern markers."""
    print(f"\nUnique digit sequences found: {len(unique_results)}")
    print()

    for row in unique_results[:40]:
        digits = str(row["digits"])
        groups = " ".join(digits[i : i + 4] for i in range(0, len(digits), 4))
        luhn = "LUHN-OK" if luhn_valid(digits) else ""
        has_prefix = "4388" in digits[:8]
        has_suffix = "0665" in digits[-8:]
        markers: list[str] = []

        if luhn:
            markers.append(luhn)
        if has_prefix:
            markers.append("HAS-PREFIX")
        if has_suffix:
            markers.append("HAS-SUFFIX")

        marker_str = f"  [{', '.join(markers)}]" if markers else ""
        print(
            f"  {int(row['digit_count']):2d}d | {groups:24s} | "
            f"{str(row['source']):18s} | {str(row['threshold']):15s} | "
            f"{str(row['mode']):20s}{marker_str}"
        )

    best = [
        row
        for row in unique_results
        if int(row["digit_count"]) == 16
        and "4388" in str(row["digits"])[:8]
        and "0665" in str(row["digits"])[-8:]
    ]

    if best:
        print("\n" + "=" * 70)
        print(">>> BEST CANDIDATES (16 digits, matching prefix+suffix):")
        for row in best:
            digits = str(row["digits"])
            groups = f"{digits[:4]} {digits[4:8]} {digits[8:12]} {digits[12:16]}"
            luhn_ok = luhn_valid(digits)
            print(f"    {groups}  {'LUHN-VALID' if luhn_ok else 'LUHN-FAIL'}")

    partials = [
        row
        for row in unique_results
        if ("4388" in str(row["digits"]) or "0665" in str(row["digits"]))
        and row not in best
    ]
    if partials:
        print("\n>>> PARTIAL MATCHES (contain known digit groups):")
        for row in partials[:20]:
            digits = str(row["digits"])
            print(f"    {digits:20s} | {str(row['source']):18s} | {row['mode']}")


def main() -> None:
    """Run targeted card digit extraction pipeline."""
    if len(sys.argv) < 2:
        print("Usage: python3 scripts/extract_card_digits.py <image_path>")
        sys.exit(1)

    path = sys.argv[1]
    img = load_and_inspect(path)

    print("\n=== Step 1: Remove dark occlusion (marker/redaction) ===")
    reconstructor = PixelReconstructor()

    cleaned_redact = reconstructor.remove_redactions(img)
    cv2.imwrite("/tmp/card_step1a_redact_removed.png", cleaned_redact)

    cleaned_dark = remove_dark_occlusion(img)
    cv2.imwrite("/tmp/card_step1b_dark_removed.png", cleaned_dark)

    cleaned_overlay = reconstructor.remove_color_overlay(img)
    cv2.imwrite("/tmp/card_step1c_overlay_removed.png", cleaned_overlay)

    print("\n=== Step 2: Extract card number region ===")
    sources = {
        "original": img,
        "redact-removed": cleaned_redact,
        "dark-removed": cleaned_dark,
        "overlay-removed": cleaned_overlay,
    }

    all_results: list[dict[str, Any]] = []
    for source_name, source_img in sources.items():
        process_source(source_name, source_img, all_results)

    process_full_image(img, all_results)
    process_depixelated(reconstructor, cleaned_dark, all_results)

    print("\n" + "=" * 70)
    print("=== DIGIT EXTRACTION RESULTS ===")
    print("=" * 70)

    if not all_results:
        print("No digit sequences >= 4 digits found.")
        return

    unique_results = dedupe_results(all_results)
    print_results(unique_results)


if __name__ == "__main__":
    main()
