#!/usr/bin/env python3
"""Absolute minimum OCR for POS 8 & 12 analysis. ~20 total Tesseract calls."""
import re, sys
from collections import Counter
import cv2, numpy as np, pytesseract

IMG = sys.argv[1] if len(sys.argv) > 1 else "/Users/jenineferderas/Desktop/card_image.jpg"
WL = "-c tessedit_char_whitelist=0123456789"

img = cv2.imread(IMG)
h, w = img.shape[:2]
y0, y1 = int(h*0.25), int(h*0.65)
row = img[y0:y1]
gray = cv2.cvtColor(row, cv2.COLOR_BGR2GRAY)
rh, rw = gray.shape

# Build 5 key variants
variants = []
for clip in [8, 32, 64]:
    c = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(3,3))
    variants.append((f"c{clip}i", cv2.bitwise_not(c.apply(gray))))
kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
variants.append(("tophat", cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kern)))
lut = np.array([((i/255.0)**0.3)*255 for i in range(256)]).astype(np.uint8)
variants.append(("g03i", cv2.bitwise_not(cv2.LUT(gray, lut))))

print(f"Row: {rw}x{rh}, {len(variants)} variants\n")

# Full-row reads (5 variants × 2 psm × 2 scale = 20 calls)
all_reads = []
for label, src in variants:
    for scale in [2, 3]:
        big = cv2.resize(src, (rw*scale, rh*scale), interpolation=cv2.INTER_CUBIC)
        for psm in [13, 7]:
            try:
                txt = pytesseract.image_to_string(big, config=f"--oem 3 --psm {psm} {WL}").strip()
                d = re.sub(r'\D', '', txt)
                if len(d) >= 4:
                    all_reads.append(d)
                    print(f"  [{label}/x{scale}/psm{psm}] {d}")
            except Exception:
                pass

# Also suffix zone (right 35%) — 5 variants × 2 configs = 10 calls
print("\nSuffix zone:")
sz = gray[:, int(rw*0.65):]
szh, szw = sz.shape
for label, src in variants:
    crop = src[:, int(rw*0.65):]  # same enhancement, cropped
    big = cv2.resize(crop, (crop.shape[1]*4, crop.shape[0]*4), interpolation=cv2.INTER_CUBIC)
    for psm in [7, 13]:
        try:
            txt = pytesseract.image_to_string(big, config=f"--oem 3 --psm {psm} {WL}").strip()
            d = re.sub(r'\D', '', txt)
            if len(d) >= 2:
                all_reads.append("S:" + d)
                print(f"  [{label}/psm{psm}] {d}")
        except Exception:
            pass

# === Analysis ===
full = [r for r in all_reads if not r.startswith("S:")]
suffix = [r[2:] for r in all_reads if r.startswith("S:")]

print(f"\n{'='*50}")
print(f"Total full-row reads: {len(full)}")
print(f"Total suffix reads: {len(suffix)}")

# POS 12: digit before 665
print(f"\n{'='*50}")
print("POS 12 — digit immediately before '665'")
print('='*50)
votes12 = Counter()
for r in full + suffix:
    idx = r.rfind("665")
    if idx >= 1:
        votes12[r[idx-1]] += 1

if votes12:
    total = sum(votes12.values())
    for d, n in votes12.most_common(6):
        print(f"  '{d}': {n}/{total} = {n/total:.1%}")

# Also: what 4-digit suffix patterns appear?
print("\nLast-4-digit patterns:")
last4 = Counter()
for r in full + suffix:
    if len(r) >= 4:
        last4[r[-4:]] += 1
for seq, cnt in last4.most_common(10):
    print(f"  '{seq}' x{cnt}")

# X665 patterns
print("\n?665 patterns (digit + 665):")
x665 = Counter()
for r in full + suffix:
    idx = r.rfind("665")
    if idx >= 1:
        x665[r[idx-1:idx+3]] += 1
for seq, cnt in x665.most_common(10):
    print(f"  '{seq}' x{cnt}")

# POS 8: 3rd char after 438854 (index 8 in full PAN)
print(f"\n{'='*50}")
print("POS 8 — 3rd character after '438854'")
print('='*50)
votes8 = Counter()
for r in full:
    idx = r.find("438854")
    if idx >= 0 and len(r) > idx + 8:
        votes8[r[idx+8]] += 1

if votes8:
    total = sum(votes8.values())
    for d, n in votes8.most_common(6):
        print(f"  '{d}': {n}/{total} = {n/total:.1%}")
else:
    print("  Not enough reads with >8 chars after prefix!")

# Show reads that have both 438854 and 665
print("\nReads with both '438854' AND '665':")
for r in full:
    if "438854" in r and "665" in r:
        pi = r.find("438854")
        si = r.rfind("665")
        pan = r[pi:si+3]
        mid = pan[6:-3] if len(pan) > 9 else ""
        print(f"  '{pan}'  middle='{mid}'  (len={len(pan)})")

# Show ALL unique reads
print(f"\nALL unique full-row reads ({len(set(full))}):")
for seq, cnt in Counter(full).most_common(30):
    tags = []
    if "438854" in seq: tags.append("PREFIX")
    if "665" in seq: tags.append("665")
    if "0665" in seq: tags.append("0665")
    print(f"  '{seq}' x{cnt}  {'  '.join(tags)}")
