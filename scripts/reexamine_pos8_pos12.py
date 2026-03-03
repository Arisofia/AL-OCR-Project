#!/usr/bin/env python3
"""
Re-examine PAN positions 8 and 12 with deep pixel analysis.

PAN layout (0-indexed):
  0123 4567 89AB CDEF
  4388 54?? ???? 0665
         ^       ^
        pos8    pos12

Position 8  = 3rd hidden digit (group 3, 1st char)
Position 12 = 1st digit of last group (assumed '0' — verify!)

Strategy:
  1. Column intensity profile to find actual digit x-positions
  2. Per-digit extraction at multiple scales + enhancements
  3. Vote consolidation
"""

import re
import sys
from collections import Counter

import cv2
import numpy as np
import pytesseract

IMG = sys.argv[1] if len(sys.argv) > 1 else "/Users/jenineferderas/Desktop/card_image.jpg"

OCR_WL = "-c tessedit_char_whitelist=0123456789"
OCR_EXCEPTIONS = (pytesseract.TesseractError, RuntimeError, TypeError, ValueError)


def ocr_zone(gray_zone, scale=8):
    """OCR a single-digit zone with multiple enhancements. Returns Counter of digits."""
    votes = Counter()
    h, w = gray_zone.shape[:2]

    def up(img):
        return cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)

    # Build enhancement variants
    variants = []

    # CLAHE variants
    for clip in [4, 8, 16, 32, 64]:
        c = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(3, 3))
        enh = c.apply(gray_zone)
        variants.append((f"clahe{clip}-inv", up(cv2.bitwise_not(enh))))
        variants.append((f"clahe{clip}", up(enh)))

    # Top-hat (reveals bright embossing on dark bg)
    for k in [5, 7, 11]:
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        th = cv2.morphologyEx(gray_zone, cv2.MORPH_TOPHAT, kern)
        variants.append((f"tophat{k}", up(th)))

    # Morph gradient
    kernel = np.ones((3, 3), np.uint8)
    variants.append(("morphgrad", up(cv2.morphologyEx(gray_zone, cv2.MORPH_GRADIENT, kernel))))

    # Histogram eq
    he = cv2.equalizeHist(gray_zone)
    variants.append(("histeq-inv", up(cv2.bitwise_not(he))))
    variants.append(("histeq", up(he)))

    # Gamma correction
    for gamma in [0.3, 0.5, 2.0]:
        lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype(np.uint8)
        g = cv2.LUT(gray_zone, lut)
        variants.append((f"gamma{gamma}-inv", up(cv2.bitwise_not(g))))
        variants.append((f"gamma{gamma}", up(g)))

    # Unsharp mask
    blur = cv2.GaussianBlur(gray_zone, (0, 0), 3)
    usm = cv2.addWeighted(gray_zone, 2.5, blur, -1.5, 0)
    variants.append(("usm-inv", up(cv2.bitwise_not(usm))))

    # Sobel magnitude
    sx = cv2.Sobel(gray_zone, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray_zone, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(sx ** 2 + sy ** 2)
    mag = np.clip(mag / mag.max() * 255, 0, 255).astype(np.uint8) if mag.max() > 0 else gray_zone
    variants.append(("sobel", up(mag)))

    configs = [
        f"--oem 3 --psm 10 {OCR_WL}",
        f"--oem 3 --psm 8 {OCR_WL}",
        f"--oem 3 --psm 13 {OCR_WL}",
        f"--oem 3 --psm 7 {OCR_WL}",
    ]

    thresholds = [0, 80, 100, 120, 140, 160, 180]

    for _vname, var_img in variants:
        for cfg in configs:
            for t in thresholds:
                if t == 0:
                    src = var_img
                else:
                    _, src = cv2.threshold(var_img, t, 255, cv2.THRESH_BINARY)
                try:
                    txt = pytesseract.image_to_string(src, config=cfg).strip()
                    d = re.sub(r"\D", "", txt)
                    if d:
                        votes[d[0]] += 1
                except OCR_EXCEPTIONS:
                    pass

    return votes


def column_brightness(gray_row, smooth=5):
    """Compute per-column mean brightness with smoothing."""
    profile = gray_row.mean(axis=0)
    if smooth > 1:
        kernel = np.ones(smooth) / smooth
        profile = np.convolve(profile, kernel, mode="same")
    return profile


def find_digit_centers(profile, min_bright=20, min_gap=15):
    """Find digit center x-positions from brightness peaks in a row."""
    above = profile > min_bright
    centers = []
    in_peak = False
    start = 0
    for i, v in enumerate(above):
        if v and not in_peak:
            start = i
            in_peak = True
        elif not v and in_peak:
            c = (start + i) // 2
            if not centers or (c - centers[-1]) >= min_gap:
                centers.append(c)
            in_peak = False
    if in_peak:
        c = (start + len(above)) // 2
        if not centers or (c - centers[-1]) >= min_gap:
            centers.append(c)
    return centers


def main():
    """Run deep OCR re-examination for PAN positions 8 and 12."""
    img = cv2.imread(IMG)
    if img is None:
        sys.exit(f"Cannot load {IMG}")
    h, w = img.shape[:2]
    print(f"Image: {w}x{h}\n")

    # ── Isolate card number row ──
    y0, y1 = int(h * 0.30), int(h * 0.62)
    row = img[y0:y1]
    gray = cv2.cvtColor(row, cv2.COLOR_BGR2GRAY)
    rh, rw = gray.shape

    # ── Column brightness analysis ──
    # Use CLAHE to boost embossed text
    clahe = cv2.createCLAHE(clipLimit=16.0, tileGridSize=(4, 4))
    enh = clahe.apply(gray)

    # Top-hat to isolate bright embossing
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kern)

    print("=== Column brightness analysis (top-hat, smoothed) ===")
    profile = column_brightness(tophat, smooth=7)

    # Find peaks that likely correspond to digit strokes
    centers = find_digit_centers(profile, min_bright=8, min_gap=20)
    print(f"  Detected {len(centers)} potential digit centers: {centers}")

    # We know the PAN has 16 digits in 4 groups of 4
    # Try to identify groups by larger gaps
    if len(centers) >= 4:
        gaps = [centers[i + 1] - centers[i] for i in range(len(centers) - 1)]
        median_gap = sorted(gaps)[len(gaps) // 2] if gaps else 50
        print(f"  Inter-center gaps: {gaps}")
        print(f"  Median gap: {median_gap}")

    # ── Also use char-box approach to locate '0665' precisely ──
    print("\n=== Char-box location of suffix '0665' ===")
    best_digit_boxes = []
    best_idx_0665 = -1
    best_label = "N/A"
    best_digits_len = -1
    for psm in [6, 7, 11]:
        for src, label in [
            (cv2.bitwise_not(enh), f"clahe-inv/psm{psm}"),
            (enh, f"clahe/psm{psm}"),
        ]:
            cfg = f"--oem 3 --psm {psm}"
            try:
                data = pytesseract.image_to_boxes(src, config=cfg)
            except OCR_EXCEPTIONS:
                continue
            boxes = []
            for line in data.strip().split("\n"):
                parts = line.split()
                if len(parts) >= 5:
                    ch = parts[0]
                    x1 = int(parts[1])
                    y1_b = int(parts[2])
                    x2 = int(parts[3])
                    y2_b = int(parts[4])
                    boxes.append({
                        "ch": ch,
                        "x1": x1, "y1": rh - y2_b,
                        "x2": x2, "y2": rh - y1_b,
                    })

            digit_boxes = [b for b in boxes if b["ch"].isdigit()]
            dtext = "".join(b["ch"] for b in digit_boxes)
            if "0665" in dtext:
                print(f"  Found '0665' with {label}: digits='{dtext}'")
                idx = dtext.rfind("0665")
                # Print details of '0665' boxes
                for off in range(idx, idx + 4):
                    b = digit_boxes[off]
                    print(f"    '{b['ch']}' at x=[{b['x1']},{b['x2']}] y=[{b['y1']},{b['y2']}]")
                if len(dtext) > best_digits_len:
                    best_digit_boxes = digit_boxes
                    best_idx_0665 = idx
                    best_label = label
                    best_digits_len = len(dtext)

    if not best_digit_boxes or best_idx_0665 < 0:
        print("  WARNING: Could not locate '0665' via char boxes!")
        print("  Falling back to column-profile estimation.")
        # Rough estimate: last 4 centers are '0665'
        if len(centers) >= 4:
            suffix_centers = centers[-4:]
            pitch = (suffix_centers[-1] - suffix_centers[0]) / 3.0
        else:
            pitch = 75  # fallback
            suffix_centers = [rw - 290, rw - 215, rw - 140, rw - 65]
        # Compute positions
        x_0665_start = suffix_centers[0]
        digit_y0, digit_y1 = int(rh * 0.15), int(rh * 0.85)
        print(f"  Estimated pitch: {pitch:.1f}px")
        print(f"  Estimated '0' of 0665 at x={x_0665_start}")
    else:
        print(f"\n  Using best match: {best_label}")
        idx = best_idx_0665
        b0 = best_digit_boxes[idx]
        b5 = best_digit_boxes[idx + 3]
        pitch = (b5["x1"] - b0["x1"]) / 3.0
        x_0665_start = b0["x1"]
        digit_y0 = b0["y1"] - int((b0["y2"] - b0["y1"]) * 0.3)
        digit_y1 = b0["y2"] + int((b0["y2"] - b0["y1"]) * 0.3)
        digit_y0 = max(0, digit_y0)
        digit_y1 = min(rh, digit_y1)
        dh = b0["y2"] - b0["y1"]
        print(f"  Pitch from suffix: {pitch:.1f}px")
        print(f"  '0' of '0665' at x={b0['x1']}, digit height={dh}")

    print(f"\n  Digit row y-range in ROI: [{digit_y0}, {digit_y1}]")

    # ── Position 12: the '0' of '0665' — or is it? ──
    print("\n" + "=" * 60)
    print("=== POSITION 12 — Deep re-examination ===")
    print("=" * 60)
    print("  (We assumed this is '0'. Let's verify.)\n")

    half = pitch * 0.6
    x_center_12 = x_0665_start + pitch * 0.0  # center of '0'
    zx0 = max(0, int(x_center_12 - half))
    zx1 = min(rw, int(x_center_12 + half))

    # Also try BGR channels
    blue, green, red = row[:, :, 0], row[:, :, 1], row[:, :, 2]
    channels = {"gray": gray, "blue": blue, "green": green, "red": red}

    pos12_votes = Counter()
    for ch_name, ch_img in channels.items():
        zone = ch_img[digit_y0:digit_y1, zx0:zx1]
        if zone.size == 0:
            continue
        v = ocr_zone(zone, scale=8)
        print(f"  Channel '{ch_name}': {v.most_common(5)}")
        pos12_votes += v

    # Also try slightly shifted zones (alignment uncertainty)
    for dx in [-10, -5, 5, 10]:
        zx0s = max(0, int(x_center_12 - half + dx))
        zx1s = min(rw, int(x_center_12 + half + dx))
        zone = gray[digit_y0:digit_y1, zx0s:zx1s]
        if zone.size == 0:
            continue
        v = ocr_zone(zone, scale=8)
        pos12_votes += v

    total = sum(pos12_votes.values())
    print(f"\n  POS 12 — Total reads: {total}")
    for d, n in pos12_votes.most_common(8):
        pct = n / total * 100 if total else 0
        histogram = "█" * max(1, int(pct / 2))
        print(f"    '{d}': {n:5d} ({pct:5.1f}%)  {histogram}")

    # Save zoomed POS 12 debug images
    zone_12 = gray[digit_y0:digit_y1, zx0:zx1]
    cv2.imwrite("/tmp/pos12_raw.png", zone_12)
    c = cv2.createCLAHE(clipLimit=32.0, tileGridSize=(3, 3))
    z12_enh = cv2.bitwise_not(c.apply(zone_12))
    z12_big = cv2.resize(z12_enh, (z12_enh.shape[1] * 10, z12_enh.shape[0] * 10),
                         interpolation=cv2.INTER_CUBIC)
    cv2.imwrite("/tmp/pos12_enhanced.png", z12_big)
    # Also top-hat
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    z12_th = cv2.morphologyEx(zone_12, cv2.MORPH_TOPHAT, kern)
    z12_th_big = cv2.resize(z12_th, (z12_th.shape[1] * 10, z12_th.shape[0] * 10),
                            interpolation=cv2.INTER_CUBIC)
    cv2.imwrite("/tmp/pos12_tophat.png", z12_th_big)
    print("  Debug: /tmp/pos12_raw.png, /tmp/pos12_enhanced.png, /tmp/pos12_tophat.png")

    # ── Position 8: 3rd hidden digit ──
    print("\n" + "=" * 60)
    print("=== POSITION 8 — Deep re-examination ===")
    print("=" * 60)
    print("  (Previous evidence: '5' at 48%, '8' at 41%)\n")

    # Position 8 is the 1st char of the 3rd group
    # Distance from '0' of '0665': 4 digits + 1 group gap
    # In the PAN: ...?? ???? 0665
    #                  ^
    #                 pos8
    # pos8 is 4 positions before pos12, with a group gap between pos11 and pos12
    # So: pos8_x = pos12_x - group_gap - 3*pitch
    # Group gap is typically ~0.3-0.6 * pitch

    pos8_all_votes = Counter()

    for gap_factor in [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]:
        gap = pitch * gap_factor
        # pos8 center = '0' x - gap - 3*pitch
        # The 4 hidden digits in group3 have positions 8,9,10,11
        # pos12='0', so pos11 = pos12 - gap - 1*pitch (last of group3)
        # pos10 = pos11 - pitch, pos9 = pos10 - pitch, pos8 = pos9 - pitch
        # So pos8 = pos12 - gap - 4*pitch
        x_center_8 = x_0665_start - gap - 3.5 * pitch  # center of pos8

        zx0 = max(0, int(x_center_8 - half))
        zx1 = min(rw, int(x_center_8 + half))

        gap_votes = Counter()
        for ch_name, ch_img in channels.items():
            zone = ch_img[digit_y0:digit_y1, zx0:zx1]
            if zone.size == 0:
                continue
            v = ocr_zone(zone, scale=8)
            gap_votes += v

        if gap_votes:
            top3 = gap_votes.most_common(3)
            t = sum(gap_votes.values())
            info = ", ".join(f"'{d}'={n / t:.0%}" for d, n in top3)
            print(f"  gap={gap_factor:.1f}: center_x={x_center_8:.0f}, zone=[{zx0},{zx1}] → {info}")

        pos8_all_votes += gap_votes

    # Also try dx shifts
    for gap_factor in [0.3, 0.5]:
        gap = pitch * gap_factor
        for dx in [-15, -8, 8, 15]:
            x_center_8 = x_0665_start - gap - 3.5 * pitch + dx
            zx0 = max(0, int(x_center_8 - half))
            zx1 = min(rw, int(x_center_8 + half))
            zone = gray[digit_y0:digit_y1, zx0:zx1]
            if zone.size == 0:
                continue
            pos8_all_votes += ocr_zone(zone, scale=8)

    total = sum(pos8_all_votes.values())
    print(f"\n  POS 8 — Total reads: {total}")
    for d, n in pos8_all_votes.most_common(8):
        pct = n / total * 100 if total else 0
        histogram = "█" * max(1, int(pct / 2))
        print(f"    '{d}': {n:5d} ({pct:5.1f}%)  {histogram}")

    # Save zoomed POS 8 debug images
    gap = pitch * 0.4
    x_c8 = x_0665_start - gap - 3.5 * pitch
    zx0 = max(0, int(x_c8 - half))
    zx1 = min(rw, int(x_c8 + half))
    zone_8 = gray[digit_y0:digit_y1, zx0:zx1]
    cv2.imwrite("/tmp/pos8_raw.png", zone_8)
    c = cv2.createCLAHE(clipLimit=32.0, tileGridSize=(3, 3))
    z8_enh = cv2.bitwise_not(c.apply(zone_8))
    z8_big = cv2.resize(z8_enh, (z8_enh.shape[1] * 10, z8_enh.shape[0] * 10),
                        interpolation=cv2.INTER_CUBIC)
    cv2.imwrite("/tmp/pos8_enhanced.png", z8_big)
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    z8_th = cv2.morphologyEx(zone_8, cv2.MORPH_TOPHAT, kern)
    z8_th_big = cv2.resize(z8_th, (z8_th.shape[1] * 10, z8_th.shape[0] * 10),
                           interpolation=cv2.INTER_CUBIC)
    cv2.imwrite("/tmp/pos8_tophat.png", z8_th_big)
    print("  Debug: /tmp/pos8_raw.png, /tmp/pos8_enhanced.png, /tmp/pos8_tophat.png")

    # ── Summary ──
    print("\n" + "=" * 60)
    print("=== SUMMARY ===")
    print("=" * 60)
    print("  PAN template: 4388 54?? ???? 0665")
    print(f"  Pitch: {pitch:.1f}px")

    for label, votes in [("POS 8", pos8_all_votes), ("POS 12", pos12_votes)]:
        total = sum(votes.values())
        top5 = votes.most_common(5)
        info = ", ".join(f"'{d}'={n / total:.1%}" for d, n in top5)
        print(f"  {label}: {info}  (n={total})")


if __name__ == "__main__":
    main()
