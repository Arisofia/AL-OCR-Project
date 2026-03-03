#!/usr/bin/env python3
"""
Ultra-lean position 8 & 12 re-examination.
Uses column intensity to find real digit positions, then minimal OCR.
"""
import re, sys
from collections import Counter
import cv2, numpy as np, pytesseract

IMG = sys.argv[1] if len(sys.argv) > 1 else "/Users/jenineferderas/Desktop/card_image.jpg"
WL = "-c tessedit_char_whitelist=0123456789"

img = cv2.imread(IMG)
h, w = img.shape[:2]
print(f"Image: {w}x{h}")

# -- Number row --
y0, y1 = int(h*0.30), int(h*0.62)
row = img[y0:y1]
gray = cv2.cvtColor(row, cv2.COLOR_BGR2GRAY)
rh, rw = gray.shape
print(f"Number row: {rw}x{rh}  (y={y0}-{y1})")

# -- Full-row OCR with best enhancements --
print("\n=== Full-row OCR reads ===")
clahe32 = cv2.createCLAHE(clipLimit=32.0, tileGridSize=(3,3))
enh = clahe32.apply(gray)
inv = cv2.bitwise_not(enh)

# Upscale 3x for better Tesseract accuracy
inv3 = cv2.resize(inv, (rw*3, rh*3), interpolation=cv2.INTER_CUBIC)
enh3 = cv2.resize(enh, (rw*3, rh*3), interpolation=cv2.INTER_CUBIC)

kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
th = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kern)
th3 = cv2.resize(th, (rw*3, rh*3), interpolation=cv2.INTER_CUBIC)

# Gamma 0.3
lut = np.array([((i/255.0)**0.3)*255 for i in range(256)]).astype(np.uint8)
gam = cv2.bitwise_not(cv2.LUT(gray, lut))
gam3 = cv2.resize(gam, (rw*3, rh*3), interpolation=cv2.INTER_CUBIC)

reads = Counter()
for label, src in [("clahe-inv", inv3), ("clahe", enh3), ("tophat", th3), ("gamma", gam3)]:
    for psm in [7, 13, 6]:
        for t in [0, 120, 150]:
            s = src if t == 0 else cv2.threshold(src, t, 255, cv2.THRESH_BINARY)[1]
            try:
                txt = pytesseract.image_to_string(s, config=f"--oem 3 --psm {psm} {WL}").strip()
                d = re.sub(r'\D', '', txt)
                if len(d) >= 8:
                    reads[d] += 1
                    # First time seeing this read
                    if reads[d] == 1:
                        print(f"  [{label}/psm{psm}/t{t}] {d}")
            except Exception:
                pass

# -- Analyze reads for POS 12 --
print(f"\n=== Analyzing {len(reads)} unique full-row reads ===")
print("Looking at what appears before '665' suffix:\n")

before_665 = Counter()
for seq, count in reads.items():
    idx = seq.rfind("665")
    if idx >= 1:
        digit_before = seq[idx-1]
        before_665[digit_before] += count
        # Also show 2-3 digits before
        context = seq[max(0,idx-4):idx+3]
        if count >= 1:
            print(f"  ...{context}... (digit before 665 = '{digit_before}')  x{count}")

if before_665:
    total = sum(before_665.values())
    print(f"\nDigit immediately before '665' (= POS 12?):")
    for d, n in before_665.most_common(5):
        print(f"  '{d}': {n}/{total} = {n/total:.0%}")

# -- Now look at the RIGHT half more carefully --
print("\n=== Right-half focused OCR ===")
right = gray[:, rw//2:]
rrh, rrw = right.shape

right_reads = Counter()
for clip in [8, 32]:
    c = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(3,3))
    e = c.apply(right)
    for src_img, lab in [(cv2.bitwise_not(e), f"c{clip}-inv"), (e, f"c{clip}")]:
        big = cv2.resize(src_img, (rrw*4, rrh*4), interpolation=cv2.INTER_CUBIC)
        for psm in [7, 13]:
            for t in [0, 130]:
                s = big if t == 0 else cv2.threshold(big, t, 255, cv2.THRESH_BINARY)[1]
                try:
                    txt = pytesseract.image_to_string(s, config=f"--oem 3 --psm {psm} {WL}").strip()
                    d = re.sub(r'\D', '', txt)
                    if len(d) >= 3:
                        right_reads[d] += 1
                        if right_reads[d] == 1:
                            print(f"  [{lab}/psm{psm}/t{t}] {d}")
                except Exception:
                    pass

# Also try the right 35% (suffix group only)
print("\n=== Suffix zone (right 35%) ===")
suffix_zone = gray[:, int(rw*0.65):]
szh, szw = suffix_zone.shape
suffix_reads = Counter()
for clip in [8, 16, 32, 64]:
    c = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(3,3))
    e = c.apply(suffix_zone)
    for src_img, lab in [(cv2.bitwise_not(e), f"c{clip}i"), (e, f"c{clip}")]:
        big = cv2.resize(src_img, (szw*5, szh*5), interpolation=cv2.INTER_CUBIC)
        for psm in [7, 8, 13]:
            for t in [0, 120, 150]:
                s = big if t == 0 else cv2.threshold(big, t, 255, cv2.THRESH_BINARY)[1]
                try:
                    txt = pytesseract.image_to_string(s, config=f"--oem 3 --psm {psm} {WL}").strip()
                    d = re.sub(r'\D', '', txt)
                    if len(d) >= 2:
                        suffix_reads[d] += 1
                        if suffix_reads[d] == 1:
                            print(f"  [{lab}/psm{psm}/t{t}] {d}")
                except Exception:
                    pass

print(f"\n  Unique suffix reads: {len(suffix_reads)}")
# Show top suffix reads
for seq, cnt in suffix_reads.most_common(10):
    print(f"    '{seq}' x{cnt}")

# -- Position 8: use the middle zone (between prefix and suffix) --
print("\n=== Middle zone (30%-70%) for position 8 ===")
mid_zone = gray[:, int(rw*0.30):int(rw*0.70)]
mzh, mzw = mid_zone.shape
mid_reads = Counter()
for clip in [8, 32, 64]:
    c = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(3,3))
    e = c.apply(mid_zone)
    for src_img, lab in [(cv2.bitwise_not(e), f"c{clip}i"), (e, f"c{clip}")]:
        big = cv2.resize(src_img, (mzw*4, mzh*4), interpolation=cv2.INTER_CUBIC)
        for psm in [7, 13]:
            for t in [0, 130]:
                s = big if t == 0 else cv2.threshold(big, t, 255, cv2.THRESH_BINARY)[1]
                try:
                    txt = pytesseract.image_to_string(s, config=f"--oem 3 --psm {psm} {WL}").strip()
                    d = re.sub(r'\D', '', txt)
                    if d:
                        mid_reads[d] += 1
                        if mid_reads[d] == 1:
                            print(f"  [{lab}/psm{psm}/t{t}] {d}")
                except Exception:
                    pass

# -- Summary --
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Full-row unique reads: {len(reads)}")
print(f"Right-half unique reads: {len(right_reads)}")
print(f"Suffix zone unique reads: {len(suffix_reads)}")
print(f"Middle zone unique reads: {len(mid_reads)}")

# Check: does any suffix read NOT start with 0?
print("\nSuffix reads NOT starting with '0':")
non_zero = {s: c for s, c in suffix_reads.items() if s and s[0] != '0' and len(s) >= 3}
for s, c in sorted(non_zero.items(), key=lambda x: -x[1])[:10]:
    print(f"  '{s}' x{c}")

print("\nSuffix reads starting with '0':")
zero_start = {s: c for s, c in suffix_reads.items() if s and s[0] == '0' and len(s) >= 3}
for s, c in sorted(zero_start.items(), key=lambda x: -x[1])[:10]:
    print(f"  '{s}' x{c}")

# Save debug images
cv2.imwrite("/tmp/number_row_gray.png", gray)
cv2.imwrite("/tmp/number_row_enh.png", inv)
cv2.imwrite("/tmp/suffix_zone.png", suffix_zone)
cv2.imwrite("/tmp/middle_zone.png", mid_zone)
print("\nDebug images saved to /tmp/")
