#!/usr/bin/env python3
"""
Rescan all 6 hidden positions using CORRECT coordinates from column analysis.

From card_column_analysis.py:
  - Visible suffix "0665": '0' at ~1132, pitch ~75px
  - Group gap (group3→group4): x≈1085-1131 (~47px)
  - Position 11 text visible at x=[1032,1084], center≈1058
  - Heavy marker: x=[815,1031]
  - Position 10 center ≈ 1058-75 = 983  (under marker)
  - Position 9 center ≈ 908  (under marker)
  - Position 8 center ≈ 833  (under marker)
  - Group gap (group2→group3): estimated at ~45px around x=790-833
  - Position 7 center ≈ 708  (barely visible, brightness 23)
  - Position 6 center ≈ 647  (partially visible, brightness 30)
"""
import sys, re
from collections import Counter
import cv2, numpy as np, pytesseract

img = cv2.imread("/Users/jenineferderas/Desktop/card_image.jpg")
h, w = img.shape[:2]
y0, y1 = int(h * 0.25), int(h * 0.55)
roi = img[y0:y1]
gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
print(f"Image: {w}x{h}, ROI y=[{y0},{y1}]")

PITCH = 75  # pixel width per digit (from visible suffix analysis)
# Digit centers derived from column analysis
CENTERS = {
    6: 647,
    7: 708,
    8: 833,
    9: 908,
    10: 983,
    11: 1058,
}

def enhance(g, scale=5):
    h0, w0 = g.shape
    up = lambda i: cv2.resize(i, (w0*scale, h0*scale), interpolation=cv2.INTER_CUBIC)
    out = []
    for clip in [4, 8, 16, 32, 64]:
        c = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(3,3))
        out.append(up(cv2.bitwise_not(c.apply(g))))
        out.append(up(c.apply(g)))
    k = np.ones((3,3), np.uint8)
    out.append(up(cv2.morphologyEx(g, cv2.MORPH_GRADIENT, k)))
    out.append(up(cv2.bitwise_not(cv2.equalizeHist(g))))
    for gamma in [0.3, 0.5]:
        lut = np.array([((i/255.0)**gamma)*255 for i in range(256)]).astype(np.uint8)
        out.append(up(cv2.bitwise_not(cv2.LUT(g, lut))))
    # Top-hat
    for ks in [7, 11]:
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
        out.append(up(cv2.morphologyEx(g, cv2.MORPH_TOPHAT, kern)))
    return out

cfgs = [
    "--oem 3 --psm 10 -c tessedit_char_whitelist=0123456789",
    "--oem 3 --psm 13 -c tessedit_char_whitelist=0123456789",
    "--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789",
]

def ocr1(img_var):
    results = []
    for cfg in cfgs:
        try:
            t = pytesseract.image_to_string(img_var, config=cfg).strip()
            d = re.sub(r"\D", "", t)
            if d: results.append(d[0])
        except: pass
        for th in [100, 130, 160]:
            _, bw = cv2.threshold(img_var, th, 255, cv2.THRESH_BINARY)
            try:
                t = pytesseract.image_to_string(bw, config=cfg).strip()
                d = re.sub(r"\D", "", t)
                if d: results.append(d[0])
            except: pass
    return results

channels = {"gray": gray, "blue": roi[:,:,0], "green": roi[:,:,1], "red": roi[:,:,2]}

print("\n=== Per-position scan (corrected coordinates) ===\n")
for pos in range(6, 12):
    cx = CENTERS[pos]
    votes = Counter()
    
    # Scan with small offsets around center
    for dx in [-10, -5, 0, 5, 10]:
        actual_cx = cx + dx
        hw = PITCH * 0.5
        zx0 = max(0, int(actual_cx - hw))
        zx1 = min(w, int(actual_cx + hw))
        
        for ch_name, ch_img in channels.items():
            zone = ch_img[:, zx0:zx1]
            if zone.size == 0: continue
            variants = enhance(zone)
            for v in variants:
                for d in ocr1(v):
                    votes[d] += 1

    total = sum(votes.values())
    if total:
        ranked = votes.most_common(6)
        top_str = ", ".join(f"'{d}'={n/total*100:.0f}%" for d, n in ranked[:5])
        marker_note = ""
        if 815 <= cx <= 1031:
            marker_note = " [under marker]"
        elif cx < 815 and cx > 700:
            marker_note = " [barely visible]"
        elif 1032 <= cx <= 1084:
            marker_note = " [edge visible]"
        print(f"  POS {pos} (cx={cx}){marker_note}: {top_str}  (n={total})")
    else:
        print(f"  POS {pos} (cx={cx}): no reads")

# ── Pair reads for positions 10+11 ──
print("\n=== PAIR: Positions 10+11 together ===")
pair_votes = Counter()
for dx in [-15, -5, 5, 15]:
    pair_cx = (CENTERS[10] + CENTERS[11]) / 2 + dx
    hw = PITCH * 1.1
    zx0 = max(0, int(pair_cx - hw))
    zx1 = min(w, int(pair_cx + hw))
    
    for ch_name, ch_img in channels.items():
        zone = ch_img[:, zx0:zx1]
        if zone.size == 0: continue
        variants = enhance(zone)
        for v in variants:
            for cfg in [
                "--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789",
                "--oem 3 --psm 13 -c tessedit_char_whitelist=0123456789",
            ]:
                try:
                    t = pytesseract.image_to_string(v, config=cfg).strip()
                    d = re.sub(r"\D", "", t)
                    if len(d) == 2: pair_votes[d] += 1
                except: pass
                for th in [120, 160]:
                    _, bw = cv2.threshold(v, th, 255, cv2.THRESH_BINARY)
                    try:
                        t = pytesseract.image_to_string(bw, config=cfg).strip()
                        d = re.sub(r"\D", "", t)
                        if len(d) == 2: pair_votes[d] += 1
                    except: pass

total = sum(pair_votes.values())
if total:
    for p, n in pair_votes.most_common(10):
        print(f"  '{p}': {n:4d} ({n/total*100:5.1f}%)")

# ── Group 3 as a whole (positions 8-11) ──
print("\n=== GROUP 3: Positions 8-11 together ===")
g3_votes = Counter()
g3_cx = (CENTERS[8] + CENTERS[11]) / 2
hw = (CENTERS[11] - CENTERS[8]) / 2 + PITCH * 0.6
zx0 = max(0, int(g3_cx - hw))
zx1 = min(w, int(g3_cx + hw))
for ch_name, ch_img in channels.items():
    zone = ch_img[:, zx0:zx1]
    if zone.size == 0: continue
    variants = enhance(zone, scale=4)
    for v in variants:
        for cfg in [
            "--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789",
            "--oem 3 --psm 13 -c tessedit_char_whitelist=0123456789",
        ]:
            try:
                t = pytesseract.image_to_string(v, config=cfg).strip()
                d = re.sub(r"\D", "", t)
                if len(d) >= 3: g3_votes[d] += 1
            except: pass
            for th in [100, 130, 160]:
                _, bw = cv2.threshold(v, th, 255, cv2.THRESH_BINARY)
                try:
                    t = pytesseract.image_to_string(bw, config=cfg).strip()
                    d = re.sub(r"\D", "", t)
                    if len(d) >= 3: g3_votes[d] += 1
                except: pass

total = sum(g3_votes.values())
if total:
    for seq, n in g3_votes.most_common(15):
        print(f"  '{seq}': {n:4d} ({n/total*100:5.1f}%)")

# ── Transition: pos 10-11 → 0665 ──
print("\n=== TRANSITION: last 2 hidden → 0665 ===")
tvotes = Counter()
zx0 = max(0, CENTERS[10] - int(PITCH * 0.5))
zx1 = 1420  # end of "5" in 0665
for ch_name, ch_img in channels.items():
    zone = ch_img[:, zx0:zx1]
    if zone.size == 0: continue
    variants = enhance(zone, scale=3)
    for v in variants:
        for cfg in [
            "--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789",
            "--oem 3 --psm 13 -c tessedit_char_whitelist=0123456789",
        ]:
            try:
                t = pytesseract.image_to_string(v, config=cfg).strip()
                d = re.sub(r"\D", "", t)
                if len(d) >= 4: tvotes[d] += 1
            except: pass

total = sum(tvotes.values())
if total:
    for seq, n in tvotes.most_common(15):
        tag = " ◄◄" if seq.endswith("0665") else (" ◄" if seq.endswith("665") else "")
        print(f"  '{seq}': {n:4d} ({n/total*100:5.1f}%){tag}")

# Save pos 11 debug image
print("\n=== Debug images ===")
cx11 = CENTERS[11]
hw = int(PITCH * 0.6)
zone11 = gray[:, max(0, cx11-hw):min(w, cx11+hw)]
cv2.imwrite("/tmp/pos11_correct_raw.png", zone11)
clahe = cv2.createCLAHE(clipLimit=32.0, tileGridSize=(3,3))
enh = cv2.bitwise_not(clahe.apply(zone11))
big = cv2.resize(enh, (enh.shape[1]*8, enh.shape[0]*8), interpolation=cv2.INTER_CUBIC)
cv2.imwrite("/tmp/pos11_correct_enh.png", big)
print("  /tmp/pos11_correct_raw.png, /tmp/pos11_correct_enh.png")

cx10 = CENTERS[10]
zone10 = gray[:, max(0, cx10-hw):min(w, cx10+hw)]
cv2.imwrite("/tmp/pos10_correct_raw.png", zone10)
enh10 = cv2.bitwise_not(clahe.apply(zone10))
big10 = cv2.resize(enh10, (enh10.shape[1]*8, enh10.shape[0]*8), interpolation=cv2.INTER_CUBIC)
cv2.imwrite("/tmp/pos10_correct_enh.png", big10)
print("  /tmp/pos10_correct_raw.png, /tmp/pos10_correct_enh.png")
