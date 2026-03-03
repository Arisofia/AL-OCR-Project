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

import sys
from pathlib import Path

import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ocr_reconstruct.modules.reconstruct import PixelReconstructor


def load_and_inspect(path: str) -> np.ndarray:
    """Load image and print basic info."""
    img = cv2.imread(path)
    if img is None:
        print(f"ERROR: Cannot load {path}")
        sys.exit(1)
    h, w = img.shape[:2]
    print(f"Loaded: {path}")
    print(f"  Size: {w}x{h}, channels: {img.shape[2] if len(img.shape) == 3 else 1}")
    return img


def extract_card_number_region(img: np.ndarray) -> np.ndarray:
    """
    Crop to the approximate card number region.
    Card numbers typically occupy the middle horizontal band,
    roughly 30%-60% vertically and 5%-95% horizontally.
    """
    h, w = img.shape[:2]
    # The card number line is typically in the upper-middle area
    # Adjust based on typical card layout
    y_start = int(h * 0.25)
    y_end = int(h * 0.65)
    x_start = int(w * 0.02)
    x_end = int(w * 0.98)
    roi = img[y_start:y_end, x_start:x_end]
    print(f"  Card number ROI: ({x_start},{y_start})-({x_end},{y_end}) = {roi.shape[1]}x{roi.shape[0]}")
    return roi


def remove_dark_occlusion(img: np.ndarray) -> np.ndarray:
    """
    Detect dark marker/redaction areas and inpaint them.
    For black markers on dark cards, use adaptive thresholding
    to find the very darkest regions (marker) vs card background.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

    # The marker is the VERY darkest region (< some threshold)
    # Card background is dark but not as dark as marker ink
    # Embossed text is lighter than card background
    _, dark_mask = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY_INV)

    # Filter for contiguous dark regions (markers, not just dark pixels)
    contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    marker_mask = np.zeros_like(dark_mask)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        # Markers are large dark blobs
        if area > 500 and w > 30 and h > 20:
            cv2.drawContours(marker_mask, [cnt], -1, 255, -1)

    # Dilate to cover marker edges
    kernel = np.ones((5, 5), np.uint8)
    marker_mask = cv2.dilate(marker_mask, kernel, iterations=2)

    ratio = float(np.count_nonzero(marker_mask)) / float(marker_mask.size or 1)
    print(f"  Dark occlusion detected: {ratio:.1%} of image")

    if ratio < 0.01:
        print("  No significant dark occlusion found")
        return img

    # Double-pass inpainting
    telea = cv2.inpaint(img, marker_mask, 7, cv2.INPAINT_TELEA)
    result = cv2.inpaint(telea, marker_mask, 3, cv2.INPAINT_NS)
    return result


def enhance_embossed_text(img: np.ndarray) -> np.ndarray:
    """
    Enhance embossed silver/grey text on dark background.
    Uses CLAHE + inversion to make text dark-on-white for OCR.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

    # CLAHE for local contrast enhancement (reveals subtle embossing)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Invert: make embossed text dark on light background
    inverted = cv2.bitwise_not(enhanced)

    # Additional sharpening
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(inverted, -1, kernel)

    return sharpened


def aggressive_digit_extraction(img: np.ndarray, label: str = "") -> list[dict]:
    """
    Run Tesseract in multiple modes optimized for digit extraction.
    Returns list of {mode, text, digits, confidence} results.
    """
    results = []

    configs = [
        ("PSM6-digits", f"--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789"),
        ("PSM7-digits", f"--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789"),
        ("PSM6-full", f"--oem 3 --psm 6"),
        ("PSM11-digits", f"--oem 3 --psm 11 -c tessedit_char_whitelist=0123456789"),
        ("PSM13-single", f"--oem 3 --psm 13 -c tessedit_char_whitelist=0123456789"),
        ("PSM4-columns", f"--oem 3 --psm 4"),
        ("PSM6-digits-nolstm", f"--oem 0 --psm 6 -c tessedit_char_whitelist=0123456789"),
    ]

    for mode_name, config in configs:
        try:
            text = pytesseract.image_to_string(img, config=config).strip()
            import re
            digits = re.sub(r"\D", "", text)
            if digits or text:
                results.append({
                    "mode": mode_name,
                    "text": text,
                    "digits": digits,
                    "digit_count": len(digits),
                })
        except Exception as e:
            results.append({"mode": mode_name, "error": str(e)})

    return results


def try_multiple_thresholds(gray: np.ndarray) -> list[np.ndarray]:
    """Generate multiple threshold variants for OCR attempts."""
    variants = []

    # Otsu
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(("otsu", otsu))

    # Adaptive mean
    adapt_mean = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                        cv2.THRESH_BINARY, 11, 2)
    variants.append(("adaptive-mean", adapt_mean))

    # Adaptive Gaussian
    adapt_gauss = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 11, 2)
    variants.append(("adaptive-gauss", adapt_gauss))

    # Fixed thresholds at various levels
    for thresh_val in [80, 100, 120, 140, 160, 180]:
        _, fixed = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
        variants.append((f"fixed-{thresh_val}", fixed))

    return variants


def upscale(img: np.ndarray, factor: int = 3) -> np.ndarray:
    """Upscale image for better OCR resolution."""
    h, w = img.shape[:2]
    return cv2.resize(img, (w * factor, h * factor), interpolation=cv2.INTER_CUBIC)


def main():
    """Run targeted card digit extraction pipeline."""
    if len(sys.argv) < 2:
        print("Usage: python3 scripts/extract_card_digits.py <image_path>")
        sys.exit(1)

    path = sys.argv[1]
    img = load_and_inspect(path)

    print("\n=== Step 1: Remove dark occlusion (marker/redaction) ===")
    reconstructor = PixelReconstructor()

    # First try repo's redaction removal
    cleaned_redact = reconstructor.remove_redactions(img)
    cv2.imwrite("/tmp/card_step1a_redact_removed.png", cleaned_redact)

    # Also try our targeted dark removal
    cleaned_dark = remove_dark_occlusion(img)
    cv2.imwrite("/tmp/card_step1b_dark_removed.png", cleaned_dark)

    # Also try repo's color overlay removal
    cleaned_overlay = reconstructor.remove_color_overlay(img)
    cv2.imwrite("/tmp/card_step1c_overlay_removed.png", cleaned_overlay)

    print("\n=== Step 2: Extract card number region ===")
    sources = {
        "original": img,
        "redact-removed": cleaned_redact,
        "dark-removed": cleaned_dark,
        "overlay-removed": cleaned_overlay,
    }

    all_results = []

    for source_name, source_img in sources.items():
        print(f"\n--- Processing: {source_name} ---")
        roi = extract_card_number_region(source_img)

        print("  Step 3: Enhance embossed text")
        enhanced = enhance_embossed_text(roi)
        cv2.imwrite(f"/tmp/card_{source_name}_enhanced.png", enhanced)

        # Upscale
        enhanced_big = upscale(enhanced, 3)
        cv2.imwrite(f"/tmp/card_{source_name}_enhanced_3x.png", enhanced_big)

        print("  Step 4: Try multiple threshold variants")
        threshold_variants = try_multiple_thresholds(enhanced_big)

        for variant_name, variant_img in threshold_variants:
            results = aggressive_digit_extraction(variant_img, f"{source_name}/{variant_name}")
            for r in results:
                if "digits" in r and len(r.get("digits", "")) >= 4:
                    r["source"] = source_name
                    r["threshold"] = variant_name
                    all_results.append(r)

    # Also try direct OCR on the whole image with various modes
    print("\n--- Processing: full-image direct ---")
    gray_full = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inverted_full = cv2.bitwise_not(gray_full)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    enhanced_full = clahe.apply(inverted_full)
    big_full = upscale(enhanced_full, 2)
    for variant_name, variant_img in try_multiple_thresholds(big_full):
        results = aggressive_digit_extraction(variant_img, f"full/{variant_name}")
        for r in results:
            if "digits" in r and len(r.get("digits", "")) >= 4:
                r["source"] = "full-image"
                r["threshold"] = variant_name
                all_results.append(r)

    # Also try the color overlay cleanup then handle_pixelation
    print("\n--- Processing: depixelated ---")
    depix = reconstructor.handle_pixelation(cleaned_dark)
    roi_depix = extract_card_number_region(depix)
    enhanced_depix = enhance_embossed_text(roi_depix)
    big_depix = upscale(enhanced_depix, 3)
    for variant_name, variant_img in try_multiple_thresholds(big_depix):
        results = aggressive_digit_extraction(variant_img, f"depix/{variant_name}")
        for r in results:
            if "digits" in r and len(r.get("digits", "")) >= 4:
                r["source"] = "depixelated"
                r["threshold"] = variant_name
                all_results.append(r)

    # Sort by digit count and report
    print("\n" + "=" * 70)
    print("=== DIGIT EXTRACTION RESULTS ===")
    print("=" * 70)

    if not all_results:
        print("No digit sequences >= 4 digits found.")
        return

    # Deduplicate by digits
    seen = set()
    unique_results = []
    for r in sorted(all_results, key=lambda x: -x.get("digit_count", 0)):
        d = r.get("digits", "")
        if d not in seen:
            seen.add(d)
            unique_results.append(r)

    # Check for known prefix/suffix
    import re

    print(f"\nUnique digit sequences found: {len(unique_results)}")
    print()

    # Import Luhn from our codebase
    try:
        from ocr_service.modules.personal_doc_extractor import _luhn_valid
    except ImportError:
        def _luhn_valid(n):
            if not n.isdigit() or not (13 <= len(n) <= 19):
                return False
            total = 0
            for i, ch in enumerate(reversed(n)):
                d = int(ch)
                if i % 2 == 1:
                    d = d * 2 - 9 if d > 4 else d * 2
                total += d
            return total % 10 == 0

    for r in unique_results[:40]:
        digits = r["digits"]
        groups = " ".join(digits[i:i+4] for i in range(0, len(digits), 4))
        luhn = "LUHN-OK" if _luhn_valid(digits) else ""
        has_prefix = "4388" in digits[:8]
        has_suffix = "0665" in digits[-8:]
        markers = []
        if luhn:
            markers.append(luhn)
        if has_prefix:
            markers.append("HAS-PREFIX")
        if has_suffix:
            markers.append("HAS-SUFFIX")
        marker_str = f"  [{', '.join(markers)}]" if markers else ""
        print(f"  {r['digit_count']:2d}d | {groups:24s} | {r['source']:18s} | {r['threshold']:15s} | {r['mode']:20s}{marker_str}")

    # Highlight best candidates (16 digits with known prefix/suffix)
    best = [r for r in unique_results
            if r["digit_count"] == 16
            and "4388" in r["digits"][:8]
            and "0665" in r["digits"][-8:]]

    if best:
        print("\n" + "=" * 70)
        print(">>> BEST CANDIDATES (16 digits, matching prefix+suffix):")
        for r in best:
            d = r["digits"]
            groups = f"{d[0:4]} {d[4:8]} {d[8:12]} {d[12:16]}"
            luhn_ok = _luhn_valid(d)
            print(f"    {groups}  {'LUHN-VALID' if luhn_ok else 'LUHN-FAIL'}")

    # Also find partial matches (contains known digits)
    partials = [r for r in unique_results
                if ("4388" in r["digits"] or "0665" in r["digits"])
                and r not in best]
    if partials:
        print("\n>>> PARTIAL MATCHES (contain known digit groups):")
        for r in partials[:20]:
            d = r["digits"]
            groups = " ".join(d[i:i+4] for i in range(0, len(d), 4))
            print(f"    {d:20s} | {r['source']:18s} | {r['mode']}")


if __name__ == "__main__":
    main()
