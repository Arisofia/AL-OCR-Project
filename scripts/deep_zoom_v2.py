#!/usr/bin/env python3
# pylint: disable=too-many-nested-blocks
"""
Locate '0665' on the card and work leftward to examine the two digits
immediately to its left (positions 10 and 11 in the PAN).

Approach:
1. Use Tesseract char-level boxes to find the exact bbox of '0','6','6','5'
2. Estimate digit pitch from visible text
3. Extract zones for the 2 preceding digit positions
4. Apply focused enhancements + OCR
"""

import re
import sys
from collections import Counter

import cv2
import numpy as np
import pytesseract

OCR_DIGIT_WHITELIST = "-c tessedit_char_whitelist=0123456789"
TESSERACT_PSM7 = f"--oem 3 --psm 7 {OCR_DIGIT_WHITELIST}"
TESSERACT_PSM8 = f"--oem 3 --psm 8 {OCR_DIGIT_WHITELIST}"
TESSERACT_PSM10 = f"--oem 3 --psm 10 {OCR_DIGIT_WHITELIST}"
TESSERACT_PSM13 = f"--oem 3 --psm 13 {OCR_DIGIT_WHITELIST}"
def load(path):
    """Load an image from disk and exit when input is invalid."""
    img = cv2.imread(path)
    if img is None:
        sys.exit(f"Cannot load {path}")
    return img


def get_char_boxes(img, config="--oem 3 --psm 6"):
    """Get per-character bounding boxes via Tesseract."""
    data = pytesseract.image_to_boxes(img, config=config)
    h = img.shape[0]
    boxes = []
    for line in data.strip().split("\n"):
        parts = line.split()
        if len(parts) >= 5:
            ch = parts[0]
            x1, y1, x2, y2 = int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])
            # Tesseract uses bottom-left origin; flip y
            boxes.append({"ch": ch, "x1": x1, "y1": h - y2, "x2": x2, "y2": h - y1})
    return boxes


def find_suffix_box(boxes, suffix="0665"):
    """Find the bounding box of the last 4 digits '0665' in char boxes."""
    chars = [b["ch"] for b in boxes]
    text = "".join(chars)
    # Find rightmost occurrence of 0665
    idx = text.rfind(suffix)
    if idx < 0:
        # Try with fuzzy: look for 665
        idx = text.rfind("665")
        if idx >= 0:
            idx -= 1  # include the digit before 665
    return idx


def focused_enhance(gray, scale=6):
    """Small set of effective enhancements for embossed text."""
    h, w = gray.shape[:2]
    out = []

    def up(img):
        return cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)

    # CLAHE
    for clip in [4, 8, 16, 32, 64]:
        c = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(3, 3))
        out.append((f"clahe{clip}", up(cv2.bitwise_not(c.apply(gray)))))
        out.append((f"clahe{clip}-raw", up(c.apply(gray))))

    # Morph gradient
    kernel = np.ones((3, 3), np.uint8)
    out.append(("morphgrad", up(cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel))))

    # Top-hat (reveal bright features on dark bg)
    for k in [7, 11]:
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        out.append((f"tophat{k}", up(cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kern))))

    # Histogram eq
    out.append(("histeq", up(cv2.bitwise_not(cv2.equalizeHist(gray)))))

    # Gamma
    for gamma in [0.3, 0.5]:
        lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype(np.uint8)
        out.append((f"gamma{gamma}", up(cv2.bitwise_not(cv2.LUT(gray, lut)))))

    return out


def ocr_digit(img):
    """OCR a single-digit zone. Returns list of (digit_str, config)."""
    configs = [
        ("psm10", TESSERACT_PSM10),
        ("psm8", TESSERACT_PSM8),
        ("psm13", TESSERACT_PSM13),
        ("psm7", TESSERACT_PSM7),
    ]
    results = []
    for mode, cfg in configs:
        try:
            txt = pytesseract.image_to_string(img, config=cfg).strip()
            d = re.sub(r"\D", "", txt)
            if d:
                results.append((d[0], mode))
        except (pytesseract.TesseractError, RuntimeError, TypeError, ValueError):
            pass
        # Also with a couple of thresholds
        for t in [100, 130, 160]:
            _, bw = cv2.threshold(img, t, 255, cv2.THRESH_BINARY)
            try:
                txt = pytesseract.image_to_string(bw, config=cfg).strip()
                d = re.sub(r"\D", "", txt)
                if d:
                    results.append((d[0], f"{mode}/t{t}"))
            except (pytesseract.TesseractError, RuntimeError, TypeError, ValueError):
                pass
    return results


def main() -> None:  # pylint: disable=too-many-locals,too-many-branches,too-many-statements,too-many-nested-blocks  # NOSONAR
    """Run focused deep-zoom OCR for PAN positions 10 and 11."""
    path = sys.argv[1] if len(sys.argv) > 1 else "/Users/jenineferderas/Desktop/card_image.jpg"
    img = load(path)
    h, w = img.shape[:2]
    print(f"Image: {w}x{h}")

    # ── Step 1: Locate visible digits via char-level boxes ──
    print("\n=== Step 1: Locating visible digits ===\n")

    # Focus on the card number region
    y0, y1 = int(h * 0.20), int(h * 0.70)
    roi = img[y0:y1]

    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Try different enhancements to get char boxes
    best_boxes = None
    best_text = ""
    for config in ["--oem 3 --psm 6", "--oem 3 --psm 7", "--oem 3 --psm 11"]:
        # Try with CLAHE
        clahe = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(4, 4))
        enhanced = clahe.apply(gray_roi)
        inv = cv2.bitwise_not(enhanced)
        for src, label in [(inv, "clahe-inv"), (enhanced, "clahe")]:
            boxes = get_char_boxes(src, config=config)
            chars = "".join(b["ch"] for b in boxes)
            digits = re.sub(r"\D", "", chars)
            if "0665" in digits and (
                best_boxes is None
                or len(digits) > len(re.sub(r"\D", "", best_text))
            ):
                best_boxes = boxes
                best_text = chars
                print(f"  Found with {label}/{config}: '{chars}'")
                # Find digit-only boxes
                digit_boxes = [b for b in boxes if b["ch"].isdigit()]
                digit_text = "".join(b["ch"] for b in digit_boxes)
                print(f"  Digits: {digit_text}")

    if best_boxes is None:
        print("  Could not locate '0665' via char boxes. Falling back to column analysis.")
        # Fallback: use the right portion of the image
        # We know the card has format 4388 54XX XXXX 0665
        # Try to find where "0665" starts by scanning from right
        # Use the right 40% of the image
        right_roi = roi[:, int(w * 0.6):]
        gray_right = (
            cv2.cvtColor(right_roi, cv2.COLOR_BGR2GRAY)
            if len(right_roi.shape) == 3
            else right_roi
        )
        clahe = cv2.createCLAHE(clipLimit=16.0, tileGridSize=(3, 3))
        enhanced_right = cv2.bitwise_not(clahe.apply(gray_right))
        boxes = get_char_boxes(enhanced_right, "--oem 3 --psm 7")
        chars = "".join(b["ch"] for b in boxes)
        print(f"  Right portion OCR: '{chars}'")
        digit_boxes = [b for b in boxes if b["ch"].isdigit()]
        if digit_boxes:
            best_boxes = digit_boxes
            # Adjust x coordinates to full-image space
            offset_x = int(w * 0.6)
            for b in best_boxes:
                b["x1"] += offset_x
                b["x2"] += offset_x

    # ── Step 2: Find the '0' of '0665' and compute digit pitch ──
    print("\n=== Step 2: Computing digit positions ===\n")

    if best_boxes:
        digit_boxes = [b for b in best_boxes if b["ch"].isdigit()]
        digit_text = "".join(b["ch"] for b in digit_boxes)
        print(f"  All digit boxes: {digit_text}")

        # Find '0665' sequence
        idx_0665 = digit_text.rfind("0665")
        if idx_0665 >= 0:
            box_0 = digit_boxes[idx_0665]
            box_6a = digit_boxes[idx_0665 + 1]
            box_6b = digit_boxes[idx_0665 + 2]
            box_5 = digit_boxes[idx_0665 + 3]

            # Digit pitch from the suffix
            pitches = []
            suffix_boxes = [box_0, box_6a, box_6b, box_5]
            for i in range(len(suffix_boxes) - 1):
                pitch = suffix_boxes[i + 1]["x1"] - suffix_boxes[i]["x1"]
                pitches.append(pitch)
            avg_pitch = sum(pitches) / len(pitches) if pitches else 30
            digit_h = box_0["y2"] - box_0["y1"]

            print(f"  '0665' starts at x={box_0['x1']} in ROI")
            print(f"  Digit pitches in suffix: {pitches}")
            print(f"  Average digit pitch: {avg_pitch:.1f}px")
            print(f"  Digit height: {digit_h}px")
            print(
                f"  '0' bbox: x=[{box_0['x1']}, {box_0['x2']}], "
                f"y=[{box_0['y1']}, {box_0['y2']}]"
            )

            # Position 11 center = 0's x1 - 0.5*pitch (one digit to the left of '0')
            # But there's a group space between group3 and group4: ???? 0665
            # The space adds roughly 0.3-0.5 * pitch
            # Position 10 = one more pitch to the left

            print("\n  Scanning multiple offsets for group gap...")

            for gap_factor in [0.0, 0.3, 0.5, 0.7, 1.0]:
                gap = avg_pitch * gap_factor
                pos11_center = box_0["x1"] - gap - avg_pitch * 0.5
                pos10_center = box_0["x1"] - gap - avg_pitch * 1.5
                print(
                    f"  gap={gap_factor:.1f}*pitch: "
                    f"pos10_center={pos10_center:.0f}, "
                    f"pos11_center={pos11_center:.0f}"
                )

            # ── Step 3: Extract and OCR positions 10 and 11 ──
            print("\n=== Step 3: Deep OCR on positions 10 & 11 ===\n")

            gray_full_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            channels = {
                "gray": gray_full_roi,
                "blue": roi[:, :, 0],
                "green": roi[:, :, 1],
                "red": roi[:, :, 2],
            }

            for pos_label, pos_idx in [("POS 11", 0.5), ("POS 10", 1.5)]:
                print(f"  ── {pos_label} ──")
                grand_votes = Counter()

                for gap_factor in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
                    gap = avg_pitch * gap_factor
                    center_x = box_0["x1"] - gap - avg_pitch * pos_idx

                    half_w = avg_pitch * 0.55
                    zx0 = max(0, int(center_x - half_w))
                    zx1 = min(w, int(center_x + half_w))

                    zy0 = max(0, box_0["y1"] - int(digit_h * 0.3))
                    zy1 = min(roi.shape[0], box_0["y2"] + int(digit_h * 0.3))

                    for ch_img in channels.values():
                        zone = ch_img[zy0:zy1, zx0:zx1]
                        if zone.size == 0:
                            continue
                        variants = focused_enhance(zone, scale=6)
                        for _, var_img in variants:
                            reads = ocr_digit(var_img)
                            for digit, _mode in reads:
                                grand_votes[digit] += 1

                if grand_votes:
                    total = sum(grand_votes.values())
                    ranked = grand_votes.most_common(8)
                    print(f"    Total reads: {total}")
                    for d, n in ranked:
                        pct = n / total * 100
                        bar_graph = "█" * max(1, int(pct / 2))
                        print(f"      '{d}': {n:4d} ({pct:5.1f}%)  {bar_graph}")
                else:
                    print("    No reads")
                print()

            # ── Step 4: Two-digit pair ──
            print("  ── PAIR (pos 10+11 together) ──")
            pair_votes = Counter()
            for gap_factor in [0.0, 0.3, 0.5, 0.7, 1.0]:
                gap = avg_pitch * gap_factor
                center_x = box_0["x1"] - gap - avg_pitch
                half_w = avg_pitch * 1.1
                zx0 = max(0, int(center_x - half_w))
                zx1 = min(w, int(center_x + half_w))
                zy0 = max(0, box_0["y1"] - int(digit_h * 0.3))
                zy1 = min(roi.shape[0], box_0["y2"] + int(digit_h * 0.3))

                for ch_img in channels.values():
                    zone = ch_img[zy0:zy1, zx0:zx1]
                    if zone.size == 0:
                        continue
                    variants = focused_enhance(zone, scale=5)
                    for _, var_img in variants:
                        for cfg in [TESSERACT_PSM7, TESSERACT_PSM8, TESSERACT_PSM13]:
                            try:
                                txt = pytesseract.image_to_string(var_img, config=cfg).strip()
                                d = re.sub(r"\D", "", txt)
                                if len(d) == 2:
                                    pair_votes[d] += 1
                            except (
                                pytesseract.TesseractError,
                                RuntimeError,
                                TypeError,
                                ValueError,
                            ):
                                pass
                            for t in [100, 140]:
                                _, bw = cv2.threshold(var_img, t, 255, cv2.THRESH_BINARY)
                                try:
                                    txt = pytesseract.image_to_string(bw, config=cfg).strip()
                                    d = re.sub(r"\D", "", txt)
                                    if len(d) == 2:
                                        pair_votes[d] += 1
                                except (
                                    pytesseract.TesseractError,
                                    RuntimeError,
                                    TypeError,
                                    ValueError,
                                ):
                                    pass

            if pair_votes:
                total = sum(pair_votes.values())
                ranked = pair_votes.most_common(10)
                print(f"    Total pair reads: {total}")
                for pair, n in ranked:
                    pct = n / total * 100
                    print(f"      '{pair}': {n:4d} ({pct:5.1f}%)")
            print()

            # ── Step 5: Transition zone (last 2-3 hidden + 0665) ──
            print("  ── TRANSITION (last hidden digits → 0665) ──")
            trans_votes = Counter()
            for gap_extra in range(-20, 21, 10):
                zx0 = max(0, int(box_0["x1"] - avg_pitch * 3 + gap_extra))
                zx1 = min(w, box_5["x2"] + 10)
                zy0 = max(0, box_0["y1"] - int(digit_h * 0.3))
                zy1 = min(roi.shape[0], box_0["y2"] + int(digit_h * 0.3))

                zone = gray_full_roi[zy0:zy1, zx0:zx1]
                if zone.size == 0:
                    continue
                variants = focused_enhance(zone, scale=4)
                for _, var_img in variants:
                    for cfg in [TESSERACT_PSM7, TESSERACT_PSM13]:
                        try:
                            txt = pytesseract.image_to_string(var_img, config=cfg).strip()
                            d = re.sub(r"\D", "", txt)
                            if len(d) >= 4:
                                trans_votes[d] += 1
                        except (
                            pytesseract.TesseractError,
                            RuntimeError,
                            TypeError,
                            ValueError,
                        ):
                            pass

            if trans_votes:
                total = sum(trans_votes.values())
                ranked = trans_votes.most_common(15)
                print(f"    Total transition reads: {total}")
                for seq, n in ranked:
                    pct = n / total * 100
                    tag = " ◄◄ SUFFIX MATCH" if seq.endswith("0665") else ""
                    tag2 = " ◄ ends 665" if seq.endswith("665") and not tag else ""
                    print(f"      '{seq}': {n:4d} ({pct:5.1f}%){tag}{tag2}")

            # ── Save zoomed images for visual inspection ──
            print("\n=== Debug images saved to /tmp/ ===")
            for pos_label, pos_idx in [("pos11", 0.5), ("pos10", 1.5)]:
                for gap_factor in [0.0, 0.5]:
                    gap = avg_pitch * gap_factor
                    cx = box_0["x1"] - gap - avg_pitch * pos_idx
                    half = avg_pitch * 0.55
                    zx0 = max(0, int(cx - half))
                    zx1 = min(w, int(cx + half))
                    zy0 = max(0, box_0["y1"] - int(digit_h * 0.3))
                    zy1 = min(roi.shape[0], box_0["y2"] + int(digit_h * 0.3))
                    zone = gray_full_roi[zy0:zy1, zx0:zx1]
                    # Save raw
                    cv2.imwrite(f"/tmp/dz_{pos_label}_gap{gap_factor}_raw.png", zone)
                    # Save enhanced 8x
                    clahe = cv2.createCLAHE(clipLimit=32.0, tileGridSize=(3, 3))
                    enh = cv2.bitwise_not(clahe.apply(zone))
                    big = cv2.resize(
                        enh,
                        (enh.shape[1] * 8, enh.shape[0] * 8),
                        interpolation=cv2.INTER_CUBIC,
                    )
                    cv2.imwrite(f"/tmp/dz_{pos_label}_gap{gap_factor}_enh.png", big)
                    print(f"  /tmp/dz_{pos_label}_gap{gap_factor}_raw.png & _enh.png")
        else:
            print("  Cannot find '0665' in digit sequence. Trying partial match...")
    else:
        print("  No char boxes found at all.")


if __name__ == "__main__":
    main()
