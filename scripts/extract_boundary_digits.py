"""
Targeted OCR on marker boundary zones to extract partial digit evidence.

Focuses on the transition regions at the left and right edges of the
black marker where embossed digit outlines may still be partially visible.
"""

import sys
from pathlib import Path
import re

import cv2
import numpy as np
import pytesseract

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_image(path: str) -> np.ndarray:
    """Load and return the image."""
    img = cv2.imread(path)
    if img is None:
        print(f"ERROR: Cannot load {path}")
        sys.exit(1)
    return img


def find_marker_bounds(img: np.ndarray) -> tuple:
    """Find the horizontal bounds of the dark marker in the card number region."""
    h, w = img.shape[:2]
    y_start, y_end = int(h * 0.25), int(h * 0.65)
    roi = img[y_start:y_end, :]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi

    col_means = np.mean(gray, axis=0)
    threshold = np.percentile(col_means, 25)

    dark_cols = col_means < threshold
    dark_indices = np.nonzero(dark_cols)[0]

    if len(dark_indices) < 10:
        print("  No significant dark region found")
        return 0, w, y_start, y_end

    x_start = dark_indices[0]
    x_end = dark_indices[-1]
    print(f"  Marker bounds: x=[{x_start}, {x_end}], width={x_end - x_start}")
    print(f"  Card number Y region: [{y_start}, {y_end}]")

    return x_start, x_end, y_start, y_end


def extract_boundary_zones(
    img: np.ndarray,
    marker_x_start: int,
    marker_x_end: int,
    y_start: int,
    y_end: int,
) -> dict:
    """Extract left and right boundary zones around the marker."""
    _, width = img.shape[:2]
    margin = 80

    zones = {}

    left_x1 = max(0, marker_x_start - 120)
    left_x2 = min(width, marker_x_start + margin)
    zones["left_boundary"] = img[y_start:y_end, left_x1:left_x2]
    print(f"  Left boundary zone: x=[{left_x1}, {left_x2}]")

    right_x1 = max(0, marker_x_end - margin)
    right_x2 = min(width, marker_x_end + 120)
    zones["right_boundary"] = img[y_start:y_end, right_x1:right_x2]
    print(f"  Right boundary zone: x=[{right_x1}, {right_x2}]")

    lt_x1 = marker_x_start - 10
    lt_x2 = marker_x_start + 60
    zones["left_transition"] = img[y_start:y_end, max(0, lt_x1):min(width, lt_x2)]

    rt_x1 = marker_x_end - 60
    rt_x2 = marker_x_end + 10
    zones["right_transition"] = img[y_start:y_end, max(0, rt_x1):min(width, rt_x2)]

    marker_width = marker_x_end - marker_x_start
    digit_width = marker_width / 6
    for i in range(6):
        dx1 = int(marker_x_start + i * digit_width - 5)
        dx2 = int(marker_x_start + (i + 1) * digit_width + 5)
        zones[f"digit_pos_{i+6}"] = img[y_start:y_end, max(0, dx1):min(width, dx2)]

    return zones


def enhance_for_emboss(img: np.ndarray) -> np.ndarray:
    """Enhance image to reveal subtle embossing under/near marker."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

    clahe = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(gray)

    inverted = cv2.bitwise_not(enhanced)

    h, w = inverted.shape[:2]
    big = cv2.resize(inverted, (w * 4, h * 4), interpolation=cv2.INTER_CUBIC)

    return big


def enhance_emboss_variants(
    img: np.ndarray,
) -> list:
    """Generate multiple enhancement variants to maximize emboss visibility."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    variants = []

    for clip in [4.0, 8.0, 16.0, 32.0]:
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(4, 4))
        enhanced = clahe.apply(gray)
        inverted = cv2.bitwise_not(enhanced)
        h, w = inverted.shape[:2]
        big = cv2.resize(inverted, (w * 4, h * 4), interpolation=cv2.INTER_CUBIC)
        variants.append((f"clahe-{clip}", big))

    kernel = np.ones((3, 3), np.uint8)
    gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
    h, w = gradient.shape[:2]
    big_grad = cv2.resize(gradient, (w * 4, h * 4), interpolation=cv2.INTER_CUBIC)
    variants.append(("morph-gradient", big_grad))

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobelx**2 + sobely**2)
    sobel_scaled = (255 * sobel_mag) / (sobel_mag.max() + 1e-6)
    sobel_norm = np.clip(sobel_scaled, 0, 255).astype(np.uint8)
    h, w = sobel_norm.shape
    big_sobel = cv2.resize(sobel_norm, (w * 4, h * 4), interpolation=cv2.INTER_CUBIC)
    variants.append(("sobel-edge", big_sobel))

    equalized = cv2.equalizeHist(gray)
    inverted_eq = cv2.bitwise_not(equalized)
    h, w = inverted_eq.shape[:2]
    big_eq = cv2.resize(inverted_eq, (w * 4, h * 4), interpolation=cv2.INTER_CUBIC)
    variants.append(("histeq-inv", big_eq))

    blurred = cv2.GaussianBlur(gray, (0, 0), 5)
    unsharp = cv2.addWeighted(gray, 2.5, blurred, -1.5, 0)
    inv_unsharp = cv2.bitwise_not(unsharp)
    h, w = inv_unsharp.shape[:2]
    big_us = cv2.resize(inv_unsharp, (w * 4, h * 4), interpolation=cv2.INTER_CUBIC)
    variants.append(("unsharp-inv", big_us))

    return variants


def ocr_zone(img: np.ndarray) -> list:
    """Run multiple OCR configs on a zone image."""
    results = []
    configs = [
        ("psm7-digits", "--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789"),
        ("psm8-digits", "--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789"),
        ("psm10-digits", "--oem 3 --psm 10 -c tessedit_char_whitelist=0123456789"),
        ("psm13-digits", "--oem 3 --psm 13 -c tessedit_char_whitelist=0123456789"),
        ("psm6-digits", "--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789"),
        ("psm7-full", "--oem 3 --psm 7"),
    ]

    for mode, config in configs:
        try:
            text = pytesseract.image_to_string(img, config=config).strip()
            if text:
                digits = re.sub(r"\D", "", text)
                results.append({"mode": mode, "text": text, "digits": digits})
        except (pytesseract.TesseractError, RuntimeError, TypeError, ValueError):
            continue

    return results


def main():
    """Run boundary-zone digit extraction."""
    path = sys.argv[1] if len(sys.argv) > 1 else "/Users/jenineferderas/Desktop/card_image.jpg"
    img = load_image(path)
    h, w = img.shape[:2]
    print(f"Image: {w}x{h}")

    print("\n=== Finding marker bounds ===")
    mx_start, mx_end, my_start, my_end = find_marker_bounds(img)

    print("\n=== Extracting boundary zones ===")
    zones = extract_boundary_zones(img, mx_start, mx_end, my_start, my_end)

    print("\n=== Processing each zone with multiple enhancements ===\n")

    all_evidence = {}
    for zone_name, zone_img in zones.items():
        if zone_img.size == 0:
            continue

        zone_results = []
        variants = enhance_emboss_variants(zone_img)

        cv2.imwrite(f"/tmp/card_zone_{zone_name}.png", variants[0][1] if variants else zone_img)

        for var_name, var_img in variants:
            results = ocr_zone(var_img)
            for r in results:
                if r["digits"]:
                    r["variant"] = var_name
                    zone_results.append(r)

            for thresh in [100, 120, 140, 160]:
                _, threshed = cv2.threshold(var_img, thresh, 255, cv2.THRESH_BINARY)
                results = ocr_zone(threshed)
                for r in results:
                    if r["digits"]:
                        r["variant"] = f"{var_name}/thresh-{thresh}"
                        zone_results.append(r)

        if zone_results:
            all_evidence[zone_name] = zone_results
            unique_digits = {row["digits"] for row in zone_results if row["digits"]}
            print(
                f"  {zone_name:20s}: {len(zone_results)} reads, "
                f"unique digits: {sorted(unique_digits)[:15]}"
            )

    print("\n" + "=" * 70)
    print("=== PER-POSITION DIGIT EVIDENCE ===")
    print("=" * 70 + "\n")

    for pos in range(6, 12):
        zone_name = f"digit_pos_{pos}"
        if zone_name not in all_evidence:
            print(f"  Position {pos}: NO EVIDENCE (fully occluded)")
            continue

        digit_votes = {}
        for r in all_evidence[zone_name]:
            for d in r["digits"]:
                digit_votes[d] = digit_votes.get(d, 0) + 1

        if digit_votes:
            sorted_votes = sorted(digit_votes.items(), key=lambda x: -x[1])
            top = sorted_votes[:5]
            total = sum(v for _, v in sorted_votes)
            print(f"  Position {pos}: ", end="")
            for digit, count in top:
                pct = count / total * 100
                print(f"'{digit}'={pct:.0f}% ", end="")
            print()
        else:
            print(f"  Position {pos}: no digit reads")

    print("\n=== BOUNDARY ZONE EVIDENCE ===\n")
    for zone_name in [
        "left_boundary",
        "right_boundary",
        "left_transition",
        "right_transition",
    ]:
        if zone_name not in all_evidence:
            continue
        unique = {row["digits"] for row in all_evidence[zone_name]}
        print(f"  {zone_name}: {sorted(unique)[:10]}")


if __name__ == "__main__":
    main()
