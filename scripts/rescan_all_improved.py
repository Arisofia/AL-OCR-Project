#!/usr/bin/env python3
"""Rescan ALL hidden positions 6-11 with improved pipeline."""
import re
from collections import Counter
import cv2
import numpy as np
import pytesseract

IMG = "/Users/jenineferderas/Desktop/card_image.jpg"
WL = "-c tessedit_char_whitelist=0123456789"

img = cv2.imread(IMG)
h, w = img.shape[:2]
y0, y1 = int(h * 0.25), int(h * 0.55)
roi = img[y0:y1]
gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
rh, rw = gray.shape

# From column brightness analysis (corrected geometry)
CENTERS = {6: 647, 7: 722, 8: 833, 9: 908, 10: 983, 11: 1058}
HALF = 34
SCALE = 4
CONFIGS = [
    f"--oem 3 --psm 10 {WL}",
    f"--oem 3 --psm 13 {WL}",
    f"--oem 3 --psm 8 {WL}",
]
THRESHOLDS = [120, 150]


def upscale(im, factor=SCALE):
    return cv2.resize(im, (im.shape[1]*factor, im.shape[0]*factor),
                      interpolation=cv2.INTER_CUBIC)


def enhance_variants(zone):
    out = []
    for clip in [8, 16, 32, 64]:
        c = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(3, 3))
        out.append(cv2.bitwise_not(c.apply(zone)))
    for gamma in [0.3, 0.5]:
        lut = np.array([((i/255.0)**gamma)*255 for i in range(256)], np.uint8)
        out.append(cv2.bitwise_not(cv2.LUT(zone, lut)))
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    out.append(cv2.morphologyEx(zone, cv2.MORPH_TOPHAT, kern))
    blur = cv2.GaussianBlur(zone, (0, 0), 3)
    out.append(cv2.bitwise_not(cv2.addWeighted(zone, 2.0, blur, -1.0, 0)))
    return out


def ocr_digit(im):
    reads = []
    for cfg in CONFIGS:
        try:
            txt = pytesseract.image_to_string(im, config=cfg).strip()
            d = re.sub(r"\D", "", txt)
            if d:
                reads.append(d[0])
        except Exception:
            pass
        for thr in THRESHOLDS:
            _, bw = cv2.threshold(im, thr, 255, cv2.THRESH_BINARY)
            try:
                txt = pytesseract.image_to_string(bw, config=cfg).strip()
                d = re.sub(r"\D", "", txt)
                if d:
                    reads.append(d[0])
            except Exception:
                pass
    return reads


print(f"Image: {w}x{h}, ROI: {rw}x{rh}")
print(f"Scanning positions 6-11 with improved pipeline\n")

all_results = {}

for pos, cx in sorted(CENTERS.items()):
    votes = Counter()

    for dx in [-10, -5, -2, 0, 2, 5, 10]:
        x0 = max(0, cx + dx - HALF)
        x1 = min(rw, cx + dx + HALF)
        zone = gray[:, x0:x1]
        zone_up = upscale(zone)

        for enh in enhance_variants(zone_up):
            for d in ocr_digit(enh):
                votes[d] += 1

        # R channel
        zone_r = roi[:, x0:x1, 2]
        zone_r_up = upscale(zone_r)
        for enh in enhance_variants(zone_r_up)[:4]:
            for d in ocr_digit(enh):
                votes[d] += 1

    total = sum(votes.values())
    all_results[pos] = (votes, total)

    print(f"POS {pos} (cx={cx}): {total} reads")
    for d, n in votes.most_common(5):
        pct = n / total * 100 if total else 0
        bar = "#" * max(1, int(pct / 2))
        print(f"  '{d}': {n:4d} ({pct:5.1f}%)  {bar}")
    if votes:
        best = votes.most_common(1)[0]
        print(f"  >>> '{best[0]}' ({best[1]/total:.0%})")
    print()

# Summary table
print("=" * 50)
print("SUMMARY — Updated evidence for all hidden positions")
print("=" * 50)
print(f"{'Pos':>3}  {'#1':>8}  {'#2':>8}  {'#3':>8}  {'Total':>5}")
print("-" * 50)
for pos in range(6, 12):
    votes, total = all_results[pos]
    mc = votes.most_common(3)
    cols = []
    for d, n in mc:
        cols.append(f"'{d}'={n/total:.0%}")
    while len(cols) < 3:
        cols.append("")
    print(f"  {pos}  {cols[0]:>8}  {cols[1]:>8}  {cols[2]:>8}  {total:>5}")

# Output for easy copy into final_pan_reconstruction.py
print("\n# Python dict for WEIGHTS update:")
print("WEIGHTS = {")
for pos in range(6, 12):
    votes, total = all_results[pos]
    items = []
    for d, n in votes.most_common(5):
        items.append(f'"{d}": {n/total:.2f}')
    print(f"    {pos}: {{{', '.join(items)}}},")
print("}")

print("\n# TOP_PER_POS update:")
print("TOP_PER_POS = {")
for pos in range(6, 12):
    votes, total = all_results[pos]
    tops = "".join(d for d, _ in votes.most_common(4))
    print(f'    {pos}: "{tops}",')
print("}")
