#!/usr/bin/env python3
"""Lean re-examination of PAN positions 8 and 12. Minimal OCR calls."""

import re
import sys
from collections import Counter

import cv2
import numpy as np
import pytesseract

IMG = sys.argv[1] if len(sys.argv) > 1 else "/Users/jenineferderas/Desktop/card_image.jpg"
OCR_WL = "-c tessedit_char_whitelist=0123456789"
OCR_EXCEPTIONS = (pytesseract.TesseractError, RuntimeError, TypeError, ValueError)


def ocr_zone_fast(gray_zone, scale=6):
    """OCR a single-digit zone — lean variant."""
    votes = Counter()
    h, w = gray_zone.shape[:2]

    # 4 key enhancements only
    variants = []
    for clip in [8, 32]:
        c = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(3, 3))
        enh = c.apply(gray_zone)
        up_inv = cv2.resize(cv2.bitwise_not(enh), (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
        up_raw = cv2.resize(enh, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
        variants.append(up_inv)
        variants.append(up_raw)

    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    th = cv2.morphologyEx(gray_zone, cv2.MORPH_TOPHAT, kern)
    variants.append(cv2.resize(th, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC))

    lut = np.array([((i / 255.0) ** 0.3) * 255 for i in range(256)]).astype(np.uint8)
    gam = cv2.LUT(gray_zone, lut)
    variants.append(cv2.resize(cv2.bitwise_not(gam), (w * scale, h * scale), interpolation=cv2.INTER_CUBIC))

    configs = [
        f"--oem 3 --psm 10 {OCR_WL}",
        f"--oem 3 --psm 13 {OCR_WL}",
    ]
    thresholds = [0, 120, 160]

    for var_img in variants:
        for cfg in configs:
            for t in thresholds:
                src = var_img if t == 0 else cv2.threshold(var_img, t, 255, cv2.THRESH_BINARY)[1]
                try:
                    txt = pytesseract.image_to_string(src, config=cfg).strip()
                    d = re.sub(r"\D", "", txt)
                    if d:
                        votes[d[0]] += 1
                except OCR_EXCEPTIONS:
                    pass
    return votes


def main():
    img = cv2.imread(IMG)
    if img is None:
        sys.exit(f"Cannot load {IMG}")
    h, w = img.shape[:2]
    print(f"Image: {w}x{h}\n")

    # Card number row
    y0, y1 = int(h * 0.30), int(h * 0.62)
    row = img[y0:y1]
    gray = cv2.cvtColor(row, cv2.COLOR_BGR2GRAY)
    rh, rw = gray.shape

    # Use char boxes to find '0665'
    clahe = cv2.createCLAHE(clipLimit=16.0, tileGridSize=(4, 4))
    enh = clahe.apply(gray)

    suffix_x: float = -1.0
    pitch: float = 75.0
    digit_y0_r, digit_y1_r = int(rh * 0.15), int(rh * 0.85)

    for psm in [6, 7, 11]:
        for src in [cv2.bitwise_not(enh), enh]:
            try:
                data = pytesseract.image_to_boxes(src, config=f"--oem 3 --psm {psm}")
            except OCR_EXCEPTIONS:
                continue
            boxes = []
            for line in data.strip().split("\n"):
                parts = line.split()
                if len(parts) >= 5 and parts[0].isdigit():
                    x1, y1b, x2, y2b = int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])
                    boxes.append({"ch": parts[0], "x1": x1, "y1": rh - y2b, "x2": x2, "y2": rh - y1b})
            dtext = "".join(b["ch"] for b in boxes)
            idx = dtext.rfind("0665")
            if idx >= 0:
                b0 = boxes[idx]
                b5 = boxes[idx + 3]
                pitch = (b5["x1"] - b0["x1"]) / 3.0
                suffix_x = b0["x1"]
                digit_y0_r = max(0, b0["y1"] - int((b0["y2"] - b0["y1"]) * 0.3))
                digit_y1_r = min(rh, b0["y2"] + int((b0["y2"] - b0["y1"]) * 0.3))
                print(f"Found '0665' via psm{psm}: x={suffix_x}, pitch={pitch:.1f}")
                break
        if suffix_x >= 0:
            break

    if suffix_x < 0:
        # Fallback from column analysis
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kern)
        profile = tophat.mean(axis=0)
        k = np.ones(7) / 7
        profile = np.convolve(profile, k, mode="same")
        # Find peaks
        above = profile > 8
        centers = []
        in_pk = False
        start = 0
        for i, v in enumerate(above):
            if v and not in_pk:
                start = i
                in_pk = True
            elif not v and in_pk:
                c = (start + i) // 2
                if not centers or c - centers[-1] >= 20:
                    centers.append(c)
                in_pk = False
        if len(centers) >= 4:
            suffix_centers = centers[-4:]
            pitch = (suffix_centers[-1] - suffix_centers[0]) / 3.0
            suffix_x = suffix_centers[0]
        else:
            pitch = 75
            suffix_x = rw - 290
        print(f"Fallback: suffix_x={suffix_x}, pitch={pitch:.1f}")

    half = pitch * 0.6
    print(f"Pitch: {pitch:.1f}px, y-range: [{digit_y0_r}, {digit_y1_r}]")

    # Use gray + BGR channels
    channels = [gray, row[:, :, 0], row[:, :, 1], row[:, :, 2]]

    # ── POSITION 12 ──
    print("\n" + "=" * 50)
    print("POSITION 12 (assumed '0' of '0665')")
    print("=" * 50)

    pos12_votes = Counter()
    for dx in [-10, -5, 0, 5, 10]:
        cx = suffix_x + dx
        zx0 = max(0, int(cx - half))
        zx1 = min(rw, int(cx + half))
        for ch in channels:
            zone = ch[digit_y0_r:digit_y1_r, zx0:zx1]
            if zone.size == 0:
                continue
            pos12_votes += ocr_zone_fast(zone)

    total = sum(pos12_votes.values())
    print(f"Total reads: {total}")
    for d, n in pos12_votes.most_common(6):
        pct = n / total * 100 if total else 0
        bar = "█" * max(1, int(pct / 2))
        print(f"  '{d}': {n:4d} ({pct:5.1f}%)  {bar}")

    # ── POSITION 8 ──
    print("\n" + "=" * 50)
    print("POSITION 8 (3rd hidden digit, 1st of group 3)")
    print("=" * 50)
    print("(Previous: '5'=48%, '8'=41%)\n")

    pos8_votes = Counter()
    # pos8 = suffix_x - gap - 3*pitch (pos8 is 4th digit before '0')
    # Group gap between group3 and group4 varies
    for gap_factor in [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]:
        gap = pitch * gap_factor
        cx = suffix_x - gap - 3.5 * pitch
        for dx in [-8, 0, 8]:
            zx0 = max(0, int(cx + dx - half))
            zx1 = min(rw, int(cx + dx + half))
            for ch in channels:
                zone = ch[digit_y0_r:digit_y1_r, zx0:zx1]
                if zone.size == 0:
                    continue
                pos8_votes += ocr_zone_fast(zone)

    total = sum(pos8_votes.values())
    print(f"Total reads: {total}")
    for d, n in pos8_votes.most_common(6):
        pct = n / total * 100 if total else 0
        bar = "█" * max(1, int(pct / 2))
        print(f"  '{d}': {n:4d} ({pct:5.1f}%)  {bar}")

    # ── Also scan POS 9, 10, 11 while we're here ──
    for pos_name, pos_offset in [("POS 9 (4th hidden)", 2.5), ("POS 10 (5th hidden)", 1.5), ("POS 11 (6th hidden)", 0.5)]:
        print(f"\n{'=' * 50}")
        print(f"{pos_name}")
        print("=" * 50)
        votes = Counter()
        for gap_factor in [0.0, 0.3, 0.5, 0.7, 1.0]:
            gap = pitch * gap_factor
            cx = suffix_x - gap - pos_offset * pitch
            for dx in [-5, 0, 5]:
                zx0 = max(0, int(cx + dx - half))
                zx1 = min(rw, int(cx + dx + half))
                for ch in channels:
                    zone = ch[digit_y0_r:digit_y1_r, zx0:zx1]
                    if zone.size == 0:
                        continue
                    votes += ocr_zone_fast(zone)
        total = sum(votes.values())
        print(f"Total reads: {total}")
        for d, n in votes.most_common(6):
            pct = n / total * 100 if total else 0
            bar = "█" * max(1, int(pct / 2))
            print(f"  '{d}': {n:4d} ({pct:5.1f}%)  {bar}")

    # ── Luhn recompute ──
    print("\n" + "=" * 50)
    print("LUHN RECOMPUTE with updated evidence")
    print("=" * 50)


if __name__ == "__main__":
    main()
