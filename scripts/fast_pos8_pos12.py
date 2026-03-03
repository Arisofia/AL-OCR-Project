#!/usr/bin/env python3
"""
Fast re-examination of PAN positions 8 and 12.
Lean OCR: 5 best enhancements × 2 configs × 3 thresholds = ~30 calls/zone.
"""
import re
import sys
from collections import Counter
import cv2
import numpy as np
import pytesseract

IMG = sys.argv[1] if len(sys.argv) > 1 else "/Users/jenineferderas/Desktop/card_image.jpg"
OCR_WL = "-c tessedit_char_whitelist=0123456789"


def ocr_zone_fast(gray_zone, scale=8):
    """Fast OCR with top-5 enhancements."""
    votes = Counter()
    h, w = gray_zone.shape[:2]
    if h == 0 or w == 0:
        return votes

    def up(img):
        return cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)

    # Only the 5 best enhancements
    variants = []
    for clip in [8, 32, 64]:
        c = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(3, 3))
        enh = c.apply(gray_zone)
        variants.append(up(cv2.bitwise_not(enh)))

    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    variants.append(up(cv2.morphologyEx(gray_zone, cv2.MORPH_TOPHAT, kern)))

    lut = np.array([((i / 255.0) ** 0.3) * 255 for i in range(256)]).astype(np.uint8)
    variants.append(up(cv2.bitwise_not(cv2.LUT(gray_zone, lut))))

    configs = [f"--oem 3 --psm 10 {OCR_WL}", f"--oem 3 --psm 13 {OCR_WL}"]
    thresholds = [0, 120, 160]

    for var_img in variants:
        for cfg in configs:
            for t in thresholds:
                src = var_img if t == 0 else cv2.threshold(var_img, t, 255, cv2.THRESH_BINARY)[1]
                try:
                    txt = pytesseract.image_to_string(src, config=cfg).strip()
                    d = re.sub(r"\D", "", txt)
                    if d:
                        votes[d[0]] += 1
                except Exception:
                    pass

    return votes


def main():
    img = cv2.imread(IMG)
    if img is None:
        sys.exit(f"Cannot load {IMG}")
    h, w = img.shape[:2]
    print(f"Image: {w}x{h}")

    # Card number row
    y0, y1 = int(h * 0.30), int(h * 0.62)
    row = img[y0:y1]
    gray = cv2.cvtColor(row, cv2.COLOR_BGR2GRAY)
    rh, rw = gray.shape

    # Find '0665' via char boxes
    clahe = cv2.createCLAHE(clipLimit=16.0, tileGridSize=(4, 4))
    enh = clahe.apply(gray)

    suffix_box = None
    for psm in [6, 7, 11]:
        for src, label in [(cv2.bitwise_not(enh), "inv"), (enh, "raw")]:
            try:
                data = pytesseract.image_to_boxes(src, config=f"--oem 3 --psm {psm}")
            except Exception:
                continue
            boxes = []
            for line in data.strip().split("\n"):
                parts = line.split()
                if len(parts) >= 5:
                    boxes.append({
                        "ch": parts[0],
                        "x1": int(parts[1]), "y1": rh - int(parts[4]),
                        "x2": int(parts[3]), "y2": rh - int(parts[2]),
                    })
            dboxes = [b for b in boxes if b["ch"].isdigit()]
            dtext = "".join(b["ch"] for b in dboxes)
            idx = dtext.rfind("665")
            if idx >= 0 and (suffix_box is None or len(dtext) > suffix_box["len"]):
                suffix_box = {
                    "dboxes": dboxes, "dtext": dtext, "idx_665": idx,
                    "len": len(dtext), "label": f"{label}/psm{psm}",
                }

    if suffix_box:
        dboxes = suffix_box["dboxes"]
        idx = suffix_box["idx_665"]
        print(f"\nFound '665' via {suffix_box['label']}: digits='{suffix_box['dtext']}'")

        b6a = dboxes[idx]      # first '6'
        b6b = dboxes[idx + 1]  # second '6'
        b5 = dboxes[idx + 2]   # '5'

        pitch = ((b6b["x1"] - b6a["x1"]) + (b5["x1"] - b6b["x1"])) / 2
        digit_h = max(b6a["y2"] - b6a["y1"], b5["y2"] - b5["y1"])
        margin = int(digit_h * 0.3)
        dy0 = max(0, min(b6a["y1"], b5["y1"]) - margin)
        dy1 = min(rh, max(b6a["y2"], b5["y2"]) + margin)

        print(f"Pitch: {pitch:.1f}px, digit height: {digit_h}px, y=[{dy0},{dy1}]")

        # Position 12: one pitch LEFT of first '6' (pos 13)
        # But if there's a detected digit there, use it
        if idx > 0:
            prev = dboxes[idx - 1]
            print(f"Char-box digit before '665': '{prev['ch']}' at x=[{prev['x1']},{prev['x2']}]")
            x12_center = (prev["x1"] + prev["x2"]) / 2
        else:
            x12_center = b6a["x1"] - pitch
        print(f"POS 12 center: x={x12_center:.0f}")

        # Position 8: 4 digits + gap before pos 12
        # pos 11 through 8 are one group, separated from pos 12 by a gap
        x12 = x12_center

    else:
        # Fallback: use column brightness
        print("\nWARNING: '665' not found via char boxes. Using column profile fallback.")
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kern)
        profile = tophat.mean(axis=0)
        # Smooth
        k = np.ones(7) / 7
        profile = np.convolve(profile, k, mode="same")
        # Last significant peak cluster = suffix
        # Just use empirical: pitch ~75-95 based on image width
        pitch = 75 if rw < 1200 else 94
        x12 = rw - 4 * pitch - 20  # rough estimate
        dy0, dy1 = int(rh * 0.15), int(rh * 0.85)
        print(f"Fallback pitch: {pitch:.0f}, estimated pos12 center: {x12:.0f}")

    half = pitch * 0.6

    # ═══ POSITION 12 ═══
    print(f"\n{'='*50}")
    print("POSITION 12 (first digit of last group — assumed '0')")
    print(f"{'='*50}")

    pos12_votes = Counter()
    for dx in [-8, -4, 0, 4, 8]:
        cx = x12 + dx
        zx0 = max(0, int(cx - half))
        zx1 = min(rw, int(cx + half))
        zone = gray[dy0:dy1, zx0:zx1]
        v = ocr_zone_fast(zone)
        pos12_votes += v
        if dx == 0:
            # Also try RGB channels
            for ch in range(3):
                zone_c = row[dy0:dy1, zx0:zx1, ch]
                pos12_votes += ocr_zone_fast(zone_c)

    total = sum(pos12_votes.values())
    print(f"\nPOS 12 — {total} total reads:")
    for d, n in pos12_votes.most_common(8):
        pct = n / total * 100 if total else 0
        bar = "#" * max(1, int(pct / 2))
        print(f"  '{d}': {n:4d} ({pct:5.1f}%)  {bar}")

    # Save debug
    zx0 = max(0, int(x12 - half))
    zx1 = min(rw, int(x12 + half))
    z12 = gray[dy0:dy1, zx0:zx1]
    cv2.imwrite("/tmp/pos12_raw.png", z12)
    c = cv2.createCLAHE(clipLimit=32.0, tileGridSize=(3, 3))
    z12e = cv2.resize(cv2.bitwise_not(c.apply(z12)),
                      (z12.shape[1]*10, z12.shape[0]*10), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite("/tmp/pos12_enh10x.png", z12e)

    # ═══ POSITION 8 ═══
    print(f"\n{'='*50}")
    print("POSITION 8 (3rd hidden digit — was '5' 48% vs '8' 41%)")
    print(f"{'='*50}")

    pos8_votes = Counter()
    for gap_f in [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]:
        gap = pitch * gap_f
        # pos8 = pos12 - gap - 3.5*pitch  (center of 4th digit back from pos12)
        x8 = x12 - gap - 3.5 * pitch
        for dx in [-6, 0, 6]:
            cx = x8 + dx
            zx0 = max(0, int(cx - half))
            zx1 = min(rw, int(cx + half))
            zone = gray[dy0:dy1, zx0:zx1]
            if zone.size == 0:
                continue
            v = ocr_zone_fast(zone)
            pos8_votes += v

        if gap_f == 0.4:
            # Also try RGB channels at this gap
            zx0 = max(0, int(x8 - half))
            zx1 = min(rw, int(x8 + half))
            for ch in range(3):
                zone_c = row[dy0:dy1, zx0:zx1, ch]
                if zone_c.size > 0:
                    pos8_votes += ocr_zone_fast(zone_c)

    total = sum(pos8_votes.values())
    print(f"\nPOS 8 — {total} total reads:")
    for d, n in pos8_votes.most_common(8):
        pct = n / total * 100 if total else 0
        bar = "#" * max(1, int(pct / 2))
        print(f"  '{d}': {n:4d} ({pct:5.1f}%)  {bar}")

    # Save debug
    gap = pitch * 0.4
    x8 = x12 - gap - 3.5 * pitch
    zx0 = max(0, int(x8 - half))
    zx1 = min(rw, int(x8 + half))
    z8 = gray[dy0:dy1, zx0:zx1]
    cv2.imwrite("/tmp/pos8_raw.png", z8)
    c = cv2.createCLAHE(clipLimit=32.0, tileGridSize=(3, 3))
    z8e = cv2.resize(cv2.bitwise_not(c.apply(z8)),
                     (z8.shape[1]*10, z8.shape[0]*10), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite("/tmp/pos8_enh10x.png", z8e)

    # ═══ SUMMARY ═══
    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    d12 = pos12_votes.most_common(1)[0][0] if pos12_votes else "?"
    d8 = pos8_votes.most_common(1)[0][0] if pos8_votes else "?"

    t12 = sum(pos12_votes.values())
    t8 = sum(pos8_votes.values())
    top5_12 = ", ".join(f"'{d}'={n/t12:.0%}" for d, n in pos12_votes.most_common(5)) if t12 else "?"
    top5_8 = ", ".join(f"'{d}'={n/t8:.0%}" for d, n in pos8_votes.most_common(5)) if t8 else "?"

    print(f"  POS 12: {top5_12}")
    print(f"  POS  8: {top5_8}")
    print(f"\n  Updated PAN: 4388 54?{d8} ???? {d12}665")
    print(f"  Debug images: /tmp/pos8_*.png, /tmp/pos12_*.png")


if __name__ == "__main__":
    main()
