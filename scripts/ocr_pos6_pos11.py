"""Fast OCR sweep for positions 6 and 11 only — lean and corrected."""
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

POSITIONS = {6: 647, 11: 1058}
HALF = 34
SCALE = 4
CONFIGS = [
    f"--oem 3 --psm 10 {WL}",
    f"--oem 3 --psm 13 {WL}",
    f"--oem 3 --psm 8 {WL}",
]
THRESHOLDS = [120, 150]


def upscale(im, factor=SCALE):
    return cv2.resize(im, (im.shape[1]*factor, im.shape[0]*factor), interpolation=cv2.INTER_CUBIC)


def enhance_variants(zone):
    """Return reduced set of enhanced images."""
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
    """Run OCR, return list of single-digit reads."""
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


for pos, cx in POSITIONS.items():
    print(f"\n=== POSITION {pos} (cx={cx}) ===")
    votes = Counter()

    for dx in [-10, -5, -2, 0, 2, 5, 10]:
        x0 = max(0, cx + dx - HALF)
        x1 = min(rw, cx + dx + HALF)
        zone = gray[:, x0:x1]
        zone_up = upscale(zone)

        for enh in enhance_variants(zone_up):
            for d in ocr_digit(enh):
                votes[d] += 1

        zone_r = roi[:, x0:x1, 2]
        zone_r_up = upscale(zone_r)
        for enh in enhance_variants(zone_r_up)[:4]:
            for d in ocr_digit(enh):
                votes[d] += 1

    total = sum(votes.values())
    print(f"Total reads: {total}")
    for d, n in votes.most_common(8):
        pct = n / total * 100 if total else 0
        bar = "#" * max(1, int(pct / 2))
        print(f"  '{d}': {n:4d} ({pct:5.1f}%)  {bar}")

    if votes:
        top2 = votes.most_common(2)
        print(f"\n>>> POS {pos} = '{top2[0][0]}' ({top2[0][1]}/{total} = {top2[0][1]/total:.0%})")
        if len(top2) > 1:
            print(f"    Runner-up: '{top2[1][0]}' ({top2[1][1]}/{total} = {top2[1][1]/total:.0%})")
