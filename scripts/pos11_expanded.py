#!/usr/bin/env python3
"""Expanded POS 11 scan with wider offset range and more enhancements."""
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

CX = 1058
HALF = 37
SCALE = 4
configs = [
    f"--oem 3 --psm 10 {WL}",
    f"--oem 3 --psm 13 {WL}",
    f"--oem 3 --psm 8 {WL}",
]

votes = Counter()

def try_ocr(img_in):
    """Run OCR on one image, return digit or None."""
    results = []
    for cfg in configs:
        try:
            txt = pytesseract.image_to_string(img_in, config=cfg).strip()
            d = re.sub(r"\D", "", txt)
            if d:
                results.append(d[0])
        except Exception:
            pass
        # One threshold
        _, bw = cv2.threshold(img_in, 130, 255, cv2.THRESH_BINARY)
        try:
            txt = pytesseract.image_to_string(bw, config=cfg).strip()
            d = re.sub(r"\D", "", txt)
            if d:
                results.append(d[0])
        except Exception:
            pass
    return results

print(f"Image: {w}x{h}, ROI: {rw}x{rh}")
print(f"Scanning POS 11 at cx={CX}, half={HALF}\n")

# Multiple offsets
for dx in [-15, -10, -5, -3, 0, 3, 5, 10, 15]:
    cx = CX + dx
    x0 = max(0, cx - HALF)
    x1 = min(rw, cx + HALF)
    zone = gray[:, x0:x1]

    # Enhancement 1: CLAHE inv
    for clip in [8, 32]:
        c = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(3, 3))
        enh = cv2.bitwise_not(c.apply(zone))
        big = cv2.resize(enh, (enh.shape[1]*SCALE, enh.shape[0]*SCALE), interpolation=cv2.INTER_CUBIC)
        for d in try_ocr(big):
            votes[d] += 1

    # Enhancement 2: top-hat
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    th = cv2.morphologyEx(zone, cv2.MORPH_TOPHAT, kern)
    big_th = cv2.resize(th, (th.shape[1]*SCALE, th.shape[0]*SCALE), interpolation=cv2.INTER_CUBIC)
    for d in try_ocr(big_th):
        votes[d] += 1

    # Enhancement 3: gamma 0.3 inv
    lut = np.array([((i/255.0)**0.3)*255 for i in range(256)]).astype(np.uint8)
    gm = cv2.bitwise_not(cv2.LUT(zone, lut))
    big_gm = cv2.resize(gm, (gm.shape[1]*SCALE, gm.shape[0]*SCALE), interpolation=cv2.INTER_CUBIC)
    for d in try_ocr(big_gm):
        votes[d] += 1

# Also BGR channels at center offset
for ch_idx in range(3):
    zone_ch = roi[0:rh, max(0,CX-HALF):min(rw,CX+HALF), ch_idx]
    for clip in [8, 32]:
        c = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(3, 3))
        enh = cv2.bitwise_not(c.apply(zone_ch))
        big = cv2.resize(enh, (enh.shape[1]*SCALE, enh.shape[0]*SCALE), interpolation=cv2.INTER_CUBIC)
        for d in try_ocr(big):
            votes[d] += 1

total = sum(votes.values())
print(f"\n=== POSITION 11 RESULTS ===")
print(f"Total reads: {total}\n")
for d, n in votes.most_common(10):
    pct = n / total * 100 if total else 0
    bar = "#" * max(1, int(pct / 2))
    print(f"  '{d}': {n:3d} ({pct:5.1f}%)  {bar}")

if votes:
    best = votes.most_common(1)[0]
    print(f"\n>>> POS 11 = '{best[0]}' ({best[1]}/{total} = {best[1]/total:.0%})")
