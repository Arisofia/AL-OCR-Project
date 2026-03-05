"""Bare minimum POS 11 scan. ~20 OCR calls total."""
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
gray = cv2.cvtColor(img[y0:y1], cv2.COLOR_BGR2GRAY)
rh, rw = gray.shape

CX = 1058
HALF = 37

votes = Counter()

for dx in [-8, 0, 8]:
    x0 = max(0, CX + dx - HALF)
    x1 = min(rw, CX + dx + HALF)
    zone = gray[:, x0:x1]
    c = cv2.createCLAHE(clipLimit=32.0, tileGridSize=(3, 3))
    enh = cv2.bitwise_not(c.apply(zone))
    big = cv2.resize(enh, (enh.shape[1]*4, enh.shape[0]*4), interpolation=cv2.INTER_CUBIC)
    for psm in [10, 13]:
        cfg = f"--oem 3 --psm {psm} {WL}"
        try:
            txt = pytesseract.image_to_string(big, config=cfg).strip()
            d = re.sub(r"\D", "", txt)
            if d:
                votes[d[0]] += 1
        except Exception:
            pass
    _, bw = cv2.threshold(big, 130, 255, cv2.THRESH_BINARY)
    for psm in [10, 13]:
        cfg = f"--oem 3 --psm {psm} {WL}"
        try:
            txt = pytesseract.image_to_string(bw, config=cfg).strip()
            d = re.sub(r"\D", "", txt)
            if d:
                votes[d[0]] += 1
        except Exception:
            pass

zone = gray[:, max(0, CX-HALF):min(rw, CX+HALF)]
kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
th = cv2.morphologyEx(zone, cv2.MORPH_TOPHAT, kern)
big_th = cv2.resize(th, (th.shape[1]*4, th.shape[0]*4), interpolation=cv2.INTER_CUBIC)
for psm in [10, 13]:
    cfg = f"--oem 3 --psm {psm} {WL}"
    try:
        txt = pytesseract.image_to_string(big_th, config=cfg).strip()
        d = re.sub(r"\D", "", txt)
        if d:
            votes[d[0]] += 1
    except Exception:
        pass

for ch in range(3):
    ch_img = img[y0:y1, :, ch]
    zone_ch = ch_img[:, max(0, CX-HALF):min(rw, CX+HALF)]
    c = cv2.createCLAHE(clipLimit=32.0, tileGridSize=(3, 3))
    enh = cv2.bitwise_not(c.apply(zone_ch))
    big = cv2.resize(enh, (enh.shape[1]*4, enh.shape[0]*4), interpolation=cv2.INTER_CUBIC)
    cfg = f"--oem 3 --psm 10 {WL}"
    try:
        txt = pytesseract.image_to_string(big, config=cfg).strip()
        d = re.sub(r"\D", "", txt)
        if d:
            votes[d[0]] += 1
    except Exception:
        pass

total = sum(votes.values())
print(f"POS 11 (cx={CX}) — {total} reads:")
for d, n in votes.most_common(8):
    pct = n / total * 100 if total else 0
    print(f"  '{d}': {n} ({pct:.0f}%)")

z = gray[:, max(0, CX-HALF):min(rw, CX+HALF)]
cv2.imwrite("/tmp/p11_raw.png", z)
c = cv2.createCLAHE(clipLimit=32.0, tileGridSize=(3, 3))
ze = cv2.bitwise_not(c.apply(z))
zb = cv2.resize(ze, (ze.shape[1]*10, ze.shape[0]*10), interpolation=cv2.INTER_CUBIC)
cv2.imwrite("/tmp/p11_enh.png", zb)
print("Debug: /tmp/p11_raw.png, /tmp/p11_enh.png")
