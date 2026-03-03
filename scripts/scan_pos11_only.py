#!/usr/bin/env python3
"""
Ultra-focused scan of PAN position 11 only.

Layout: 4388 54?? ???? 0665
                    ^
                   pos11 (last hidden digit before group gap + '0665')

From column analysis:
  - pos11 center x=1058 (edge-visible, brightness ~28)
  - pitch=75, y_roi = 25%-55% of image height
  - Assuming pos12='0' is confirmed

MINIMAL OCR: 3 enhancements × 2 configs × 2 thresholds = 12 calls per zone × 4 channels × 7 offsets ≈ 336 total calls
"""
import re
import sys
from collections import Counter
import cv2
import numpy as np
import pytesseract

IMG = "/Users/jenineferderas/Desktop/card_image.jpg"
POS11_CX = 1058
PITCH = 75
OCR_WL = "-c tessedit_char_whitelist=0123456789"


def main():
    img = cv2.imread(IMG)
    if img is None:
        sys.exit(f"Cannot load {IMG}")
    h, w = img.shape[:2]

    # ROI
    y0, y1 = int(h * 0.25), int(h * 0.55)
    roi = img[y0:y1]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    rh, rw = gray.shape
    print(f"Image: {w}x{h}, ROI: {rw}x{rh}")

    channels = [gray, roi[:, :, 0], roi[:, :, 1], roi[:, :, 2]]
    ch_names = ["gray", "blue", "green", "red"]

    half = int(PITCH * 0.55)
    scale = 6

    configs = [
        f"--oem 3 --psm 10 {OCR_WL}",
        f"--oem 3 --psm 13 {OCR_WL}",
    ]
    thresholds = [0, 130]

    def up(zone):
        zh, zw = zone.shape[:2]
        return cv2.resize(zone, (zw * scale, zh * scale), interpolation=cv2.INTER_CUBIC)

    def make_variants(zone):
        variants = []
        # CLAHE inv (best for embossed text)
        for clip in [8, 32]:
            c = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(3, 3))
            variants.append(up(cv2.bitwise_not(c.apply(zone))))
        # Top-hat (bright features on dark bg)
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        variants.append(up(cv2.morphologyEx(zone, cv2.MORPH_TOPHAT, kern)))
        return variants

    votes = Counter()
    per_channel = {n: Counter() for n in ch_names}

    offsets = [-12, -6, -3, 0, 3, 6, 12]

    for dx in offsets:
        cx = POS11_CX + dx
        x0 = max(0, cx - half)
        x1 = min(rw, cx + half)

        for ch_img, ch_name in zip(channels, ch_names):
            zone = ch_img[0:rh, x0:x1]
            if zone.size == 0:
                continue

            for var_img in make_variants(zone):
                for cfg in configs:
                    for t in thresholds:
                        src = var_img if t == 0 else cv2.threshold(var_img, t, 255, cv2.THRESH_BINARY)[1]
                        try:
                            txt = pytesseract.image_to_string(src, config=cfg).strip()
                            d = re.sub(r"\D", "", txt)
                            if d:
                                votes[d[0]] += 1
                                per_channel[ch_name][d[0]] += 1
                        except Exception:
                            pass

    # Results
    total = sum(votes.values())
    print(f"\n=== POSITION 11 (cx={POS11_CX}) ===")
    print(f"Total reads: {total}\n")

    print("COMBINED:")
    for d, n in votes.most_common(10):
        pct = n / total * 100 if total else 0
        bar = "#" * max(1, int(pct / 2))
        print(f"  '{d}': {n:4d} ({pct:5.1f}%)  {bar}")

    for ch_name in ch_names:
        cv = per_channel[ch_name]
        ct = sum(cv.values())
        if ct:
            top3 = ", ".join(f"'{d}'={n/ct:.0%}" for d, n in cv.most_common(3))
            print(f"\n  {ch_name:5s}: {top3}  (n={ct})")

    # Save debug images
    x0 = max(0, POS11_CX - half)
    x1 = min(rw, POS11_CX + half)
    z = gray[0:rh, x0:x1]
    cv2.imwrite("/tmp/pos11_only_raw.png", z)
    zbig = cv2.resize(z, (z.shape[1]*10, z.shape[0]*10), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite("/tmp/pos11_only_10x.png", zbig)
    c = cv2.createCLAHE(clipLimit=32.0, tileGridSize=(3, 3))
    ze = cv2.bitwise_not(c.apply(z))
    zebig = cv2.resize(ze, (ze.shape[1]*10, ze.shape[0]*10), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite("/tmp/pos11_only_enh10x.png", zebig)
    # Top-hat
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    zt = cv2.morphologyEx(z, cv2.MORPH_TOPHAT, kern)
    ztbig = cv2.resize(zt, (zt.shape[1]*10, zt.shape[0]*10), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite("/tmp/pos11_only_tophat10x.png", ztbig)
    print("\nDebug: /tmp/pos11_only_*.png")

    # Final verdict
    if votes:
        best = votes.most_common(1)[0]
        runner = votes.most_common(2)[1] if len(votes) >= 2 else ("?", 0)
        print(f"\n>>> POS 11 VERDICT: '{best[0]}' ({best[1]/total:.0%}) vs '{runner[0]}' ({runner[1]/total:.0%})")


if __name__ == "__main__":
    main()
