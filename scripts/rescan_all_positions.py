#!/usr/bin/env python3
"""
Re-examine ALL 6 hidden positions with the focused approach.
Uses several possible '0665' x-offsets to compensate for position uncertainty.
"""
import sys, re
from collections import Counter
import cv2, numpy as np, pytesseract

img = cv2.imread("/Users/jenineferderas/Desktop/card_image.jpg")
h, w = img.shape[:2]
y0, y1 = int(h*0.20), int(h*0.70)
roi = img[y0:y1]
gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

# Multiple estimates for where '0' of 0665 sits
OX_CANDIDATES = [1030, 1050, 1070, 1090]
PITCH = 40.0
oy0, oy1 = 40, 160

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
    for gamma in [0.3]:
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

channels = {"gray": gray, "red": roi[:,:,2]}

# For each position 6-11, scan with multiple ox values and gap factors
for pos in range(6, 12):
    offset_from_0 = 12 - pos  # pos11=1, pos10=2, ..., pos6=6
    votes = Counter()
    
    for ox in OX_CANDIDATES:
        for gap_factor in [0.0, 0.5, 1.0]:
            gap = PITCH * gap_factor
            cx = ox - gap - PITCH * (offset_from_0 - 0.5)
            hw = PITCH * 0.6
            zx0 = max(0, int(cx - hw))
            zx1 = min(w, int(cx + hw))
            if zx1 - zx0 < 10: continue
            
            for ch_name, ch_img in channels.items():
                zone = ch_img[oy0:oy1, zx0:zx1]
                if zone.size == 0: continue
                variants = enhance(zone)
                for v in variants:
                    for d in ocr1(v):
                        votes[d] += 1

    total = sum(votes.values())
    if total:
        ranked = votes.most_common(6)
        top_str = ", ".join(f"'{d}'={n/total*100:.0f}%" for d, n in ranked[:4])
        print(f"  POS {pos}: {top_str}  (n={total})")
    else:
        print(f"  POS {pos}: no reads")
