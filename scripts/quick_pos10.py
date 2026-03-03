#!/usr/bin/env python3
"""Quick scan of POS 10 only, using fallback coordinates from v3."""
import sys, re
from collections import Counter
import cv2, numpy as np, pytesseract

img = cv2.imread("/Users/jenineferderas/Desktop/card_image.jpg")
h, w = img.shape[:2]
y0, y1 = int(h*0.20), int(h*0.70)
roi = img[y0:y1]
gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

# From v3 fallback: pitch=40, '0' of 0665 at x≈1050
# POS 10 = 2 digits left of 0665
# POS 11 = 1 digit left
pitch = 40.0
ox = 1050  # x of '0'
dh = 60    # digit height estimate
oy0, oy1 = 40, 160  # y range in ROI for digits

def enhance(g, scale=5):
    h0, w0 = g.shape
    up = lambda i: cv2.resize(i, (w0*scale, h0*scale), interpolation=cv2.INTER_CUBIC)
    out = []
    for clip in [8, 32, 64]:
        c = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(3,3))
        out.append(up(cv2.bitwise_not(c.apply(g))))
        out.append(up(c.apply(g)))
    k = np.ones((3,3), np.uint8)
    out.append(up(cv2.morphologyEx(g, cv2.MORPH_GRADIENT, k)))
    out.append(up(cv2.bitwise_not(cv2.equalizeHist(g))))
    for gamma in [0.3, 0.5]:
        lut = np.array([((i/255.0)**gamma)*255 for i in range(256)]).astype(np.uint8)
        out.append(up(cv2.bitwise_not(cv2.LUT(g, lut))))
    return out

cfgs = [
    "--oem 3 --psm 10 -c tessedit_char_whitelist=0123456789",
    "--oem 3 --psm 13 -c tessedit_char_whitelist=0123456789",
]

def ocr1(img_var):
    results = []
    for cfg in cfgs:
        try:
            t = pytesseract.image_to_string(img_var, config=cfg).strip()
            d = re.sub(r"\D", "", t)
            if d: results.append(d[0])
        except: pass
        for th in [120, 160]:
            _, bw = cv2.threshold(img_var, th, 255, cv2.THRESH_BINARY)
            try:
                t = pytesseract.image_to_string(bw, config=cfg).strip()
                d = re.sub(r"\D", "", t)
                if d: results.append(d[0])
            except: pass
    return results

channels = {"gray": gray, "blue": roi[:,:,0], "green": roi[:,:,1], "red": roi[:,:,2]}

# POS 10
print("POS 10 (2 digits left of 0665)")
votes10 = Counter()
for gap_factor in [0.0, 0.3, 0.5, 0.7, 1.0, 1.3]:
    gap = pitch * gap_factor
    cx = ox - gap - pitch * 1.5
    hw = pitch * 0.6
    zx0 = max(0, int(cx - hw))
    zx1 = min(w, int(cx + hw))
    for ch_name, ch_img in channels.items():
        zone = ch_img[oy0:oy1, zx0:zx1]
        if zone.size == 0: continue
        variants = enhance(zone)
        for v in variants:
            for d in ocr1(v):
                votes10[d] += 1

total = sum(votes10.values())
if total:
    for d, n in votes10.most_common(8):
        pct = n/total*100
        bar = "█" * max(1, int(pct/2))
        print(f"  '{d}': {n:4d} ({pct:5.1f}%)  {bar}")

# Also POS 11 for confirmation
print("\nPOS 11 (1 digit left of 0665)")
votes11 = Counter()
for gap_factor in [0.0, 0.3, 0.5, 0.7, 1.0, 1.3]:
    gap = pitch * gap_factor
    cx = ox - gap - pitch * 0.5
    hw = pitch * 0.6
    zx0 = max(0, int(cx - hw))
    zx1 = min(w, int(cx + hw))
    for ch_name, ch_img in channels.items():
        zone = ch_img[oy0:oy1, zx0:zx1]
        if zone.size == 0: continue
        variants = enhance(zone)
        for v in variants:
            for d in ocr1(v):
                votes11[d] += 1

total = sum(votes11.values())
if total:
    for d, n in votes11.most_common(8):
        pct = n/total*100
        bar = "█" * max(1, int(pct/2))
        print(f"  '{d}': {n:4d} ({pct:5.1f}%)  {bar}")

# PAIR read
print("\nPAIR (pos 10+11)")
pvotes = Counter()
for gap_factor in [0.0, 0.3, 0.5, 0.7, 1.0]:
    gap = pitch * gap_factor
    cx = ox - gap - pitch
    hw = pitch * 1.2
    zx0 = max(0, int(cx - hw))
    zx1 = min(w, int(cx + hw))
    zone = gray[oy0:oy1, zx0:zx1]
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
                if len(d) == 2:
                    pvotes[d] += 1
            except: pass
            for th in [120, 160]:
                _, bw = cv2.threshold(v, th, 255, cv2.THRESH_BINARY)
                try:
                    t = pytesseract.image_to_string(bw, config=cfg).strip()
                    d = re.sub(r"\D", "", t)
                    if len(d) == 2:
                        pvotes[d] += 1
                except: pass

total = sum(pvotes.values())
if total:
    for p, n in pvotes.most_common(10):
        pct = n/total*100
        print(f"  '{p}': {n:4d} ({pct:5.1f}%)")
