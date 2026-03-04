#!/usr/bin/env python3
"""
Minimal full-row OCR approach for positions 8 and 12.
No per-digit zone extraction — just full-row reads + analysis.
"""
import contextlib
import re, sys
from collections import Counter
import cv2, numpy as np, pytesseract

IMG = sys.argv[1] if len(sys.argv) > 1 else "/Users/jenineferderas/Desktop/card_image.jpg"
WL = "-c tessedit_char_whitelist=0123456789"
_SUFFIX_TAG = "SUFFIX:"
_MID_TAG = "MID:"

img = cv2.imread(IMG)
h, w = img.shape[:2]

# Number row
y0, y1 = int(h*0.25), int(h*0.65)
row = img[y0:y1]
gray = cv2.cvtColor(row, cv2.COLOR_BGR2GRAY)
rh, rw = gray.shape

# Build enhanced versions
variants = {}
for clip in [4, 8, 16, 32, 64]:
    c = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(3,3))
    e = c.apply(gray)
    variants[f"c{clip}i"] = cv2.bitwise_not(e)
    variants[f"c{clip}"] = e

kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
variants["tophat"] = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kern)

for gamma in [0.3, 0.5]:
    lut = np.array([((i/255.0)**gamma)*255 for i in range(256)]).astype(np.uint8)
    variants[f"g{gamma}i"] = cv2.bitwise_not(cv2.LUT(gray, lut))

blur = cv2.GaussianBlur(gray, (0,0), 3)
variants["usm"] = cv2.bitwise_not(cv2.addWeighted(gray, 2.5, blur, -1.5, 0))

# BGR channels
blue, green, red = row[:,:,0], row[:,:,1], row[:,:,2]
for ch_name, ch in [("blue", blue), ("green", green), ("red", red)]:
    c = cv2.createCLAHE(clipLimit=16.0, tileGridSize=(3,3))
    variants[f"{ch_name}_c16i"] = cv2.bitwise_not(c.apply(ch))

print(f"Image: {w}x{h}, row: {rw}x{rh}, {len(variants)} enhancements\n")

# Full-row OCR  
reads = []
for src in variants.values():
    for scale in [2, 3]:
        big = cv2.resize(src, (rw*scale, rh*scale), interpolation=cv2.INTER_CUBIC)
        for psm in [7, 13, 6]:
            for t in [0, 120, 150]:
                s = big if t == 0 else cv2.threshold(big, t, 255, cv2.THRESH_BINARY)[1]
                with contextlib.suppress(Exception):
                    txt = pytesseract.image_to_string(s, config=f"--oem 3 --psm {psm} {WL}").strip()
                    if d := re.sub(r'\D', '', txt):
                        if len(d) >= 6:
                            reads.append(d)

# Also suffix zone (right 40%)
sz = gray[:, int(rw*0.60):]
szh, szw = sz.shape
for clip in [8, 32]:
    c = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(3,3))
    e = c.apply(sz)
    for src_img in [cv2.bitwise_not(e), e]:
        big = cv2.resize(src_img, (szw*4, szh*4), interpolation=cv2.INTER_CUBIC)
        for psm in [7, 8, 13]:
            for t in [0, 130]:
                s = big if t == 0 else cv2.threshold(big, t, 255, cv2.THRESH_BINARY)[1]
                with contextlib.suppress(Exception):
                    txt = pytesseract.image_to_string(s, config=f"--oem 3 --psm {psm} {WL}").strip()
                    if d := re.sub(r'\D', '', txt):
                        if len(d) >= 2:
                            reads.append(_SUFFIX_TAG + d)

# Also middle zone (30%-70%) for hidden digits
mz = gray[:, int(rw*0.25):int(rw*0.72)]
mzh, mzw = mz.shape
for clip in [16, 64]:
    c = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(3,3))
    e = c.apply(mz)
    for src_img in [cv2.bitwise_not(e), e]:
        big = cv2.resize(src_img, (mzw*3, mzh*3), interpolation=cv2.INTER_CUBIC)
        for psm in [7, 13]:
            for t in [0, 140]:
                s = big if t == 0 else cv2.threshold(big, t, 255, cv2.THRESH_BINARY)[1]
                with contextlib.suppress(Exception):
                    txt = pytesseract.image_to_string(s, config=f"--oem 3 --psm {psm} {WL}").strip()
                    if d := re.sub(r'\D', '', txt):
                        reads.append(_MID_TAG + d)

# === Analyze ===
full_reads = [r for r in reads if not r.startswith((_SUFFIX_TAG, _MID_TAG))]
suffix_reads = [r[len(_SUFFIX_TAG):] for r in reads if r.startswith(_SUFFIX_TAG)]
mid_reads = [r[len(_MID_TAG):] for r in reads if r.startswith(_MID_TAG)]

print(f"Full-row reads: {len(full_reads)}")
print(f"Suffix-zone reads: {len(suffix_reads)}")
print(f"Middle-zone reads: {len(mid_reads)}")

# Show unique full reads containing 438854 or 665
print("\n=== Full-row reads containing '438854' ===")
has_prefix = Counter()
for r in full_reads:
    if "438854" in r:
        has_prefix[r] += 1
for seq, cnt in has_prefix.most_common(20):
    tag = ""
    if "665" in seq:
        tag = " <<< has 665"
    print(f"  '{seq}' x{cnt}{tag}")

print("\n=== Full-row reads containing '665' ===")
has_665 = Counter()
for r in full_reads:
    if "665" in r:
        has_665[r] += 1
for seq, cnt in has_665.most_common(20):
    tag = ""
    if "438854" in seq:
        tag = " <<< has prefix too"
    print(f"  '{seq}' x{cnt}{tag}")

# --- POS 12 analysis ---
print("\n" + "="*60)
print("POSITION 12 ANALYSIS — what digit precedes '665'?")
print("="*60)
before_665 = Counter()
for r in full_reads + suffix_reads:
    idx = r.rfind("665")
    if idx >= 1:
        before_665[r[idx-1]] += 1
if before_665:
    total = sum(before_665.values())
    for d, n in before_665.most_common(6):
        print(f"  '{d}': {n}/{total} = {n/total:.1%}")
else:
    print("  No reads containing '665' found!")

# What about reads containing '0665' vs '3665' vs '7665' etc?
print("\nFull sequence before '665':") 
for prefix_d in "0123456789":
    pattern = f"{prefix_d}665"
    count = sum(pattern in r for r in full_reads + suffix_reads)
    if count > 0:
        print(f"  '{pattern}' appears in {count} reads")

# --- POS 8 analysis ---
print("\n" + "="*60)
print("POSITION 8 ANALYSIS — 3rd hidden digit")
print("="*60)
# In 16-digit PAN 4388 54XX XXXX 0665
# Position 8 is the 1st digit of group 3
# From full reads that have both prefix and suffix, extract middle
print("Reads with both '438854' and '665':") 
full_matches = []
for r in full_reads:
    pi = r.find("438854")
    si = r.rfind("665")
    if pi >= 0 and si > pi:
        pan_portion = r[pi:si+3]  # from 438854...665
        full_matches.append(pan_portion)
        
match_counter = Counter(full_matches)
for seq, cnt in match_counter.most_common(20):
    # Extract just the hidden part
    if len(seq) >= 13:  # 438854 + X + 665 minimum
        hidden = seq[6:-3]
        print(f"  '{seq}'  hidden='{hidden}'  x{cnt}")
    else:
        print(f"  '{seq}'  (short)  x{cnt}")

# From partial reads: what's the 3rd character after '438854'?
print("\n3rd char after '438854' (= position 8):") 
pos8_votes = Counter()
for r in full_reads + mid_reads:
    idx = r.find("438854")
    if idx >= 0 and len(r) > idx + 8:
        pos8_votes[r[idx+8]] += 1
if pos8_votes:
    total = sum(pos8_votes.values())
    for d, n in pos8_votes.most_common(6):
        print(f"  '{d}': {n}/{total} = {n/total:.1%}")

# From reads containing suffix, what's 4th digit before '665'? 
# (pos 8 is 4 positions before pos12, or roughly 4th digit back from start of 0665)
print("\nMid-zone unique reads:")
mid_counter = Counter(mid_reads)
for seq, cnt in mid_counter.most_common(15):
    print(f"  '{seq}' x{cnt}")

print("\nSuffix-zone unique reads:")
suf_counter = Counter(suffix_reads)
for seq, cnt in suf_counter.most_common(15):
    print(f"  '{seq}' x{cnt}")
