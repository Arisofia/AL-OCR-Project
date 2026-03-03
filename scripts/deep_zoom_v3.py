#!/usr/bin/env python3
"""Lean deep zoom on positions 10-11. Minimal OCR calls."""
import sys, re
from collections import Counter
import cv2, numpy as np, pytesseract


def load(p):
    img = cv2.imread(p)
    if img is None: sys.exit(f"Cannot load {p}")
    return img


def char_boxes(img, cfg="--oem 3 --psm 6"):
    data = pytesseract.image_to_boxes(img, config=cfg)
    h = img.shape[0]
    boxes = []
    for line in data.strip().split("\n"):
        parts = line.split()
        if len(parts) >= 5:
            ch, x1, y1, x2, y2 = parts[0], int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])
            boxes.append({"ch": ch, "x1": x1, "y1": h - y2, "x2": x2, "y2": h - y1})
    return boxes


def enhance(gray, scale=5):
    h, w = gray.shape
    up = lambda i: cv2.resize(i, (w*scale, h*scale), interpolation=cv2.INTER_CUBIC)
    out = []
    for clip in [8, 32, 64]:
        c = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(3,3))
        out.append(up(cv2.bitwise_not(c.apply(gray))))
        out.append(up(c.apply(gray)))
    k = np.ones((3,3), np.uint8)
    out.append(up(cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, k)))
    out.append(up(cv2.bitwise_not(cv2.equalizeHist(gray))))
    for gamma in [0.3, 0.5]:
        lut = np.array([((i/255.0)**gamma)*255 for i in range(256)]).astype(np.uint8)
        out.append(up(cv2.bitwise_not(cv2.LUT(gray, lut))))
    return out


def ocr1(img):
    """Single-digit OCR. Returns list of digit chars."""
    results = []
    cfgs = [
        "--oem 3 --psm 10 -c tessedit_char_whitelist=0123456789",
        "--oem 3 --psm 13 -c tessedit_char_whitelist=0123456789",
    ]
    for cfg in cfgs:
        try:
            t = pytesseract.image_to_string(img, config=cfg).strip()
            d = re.sub(r"\D", "", t)
            if d: results.append(d[0])
        except: pass
        for th in [120, 160]:
            _, bw = cv2.threshold(img, th, 255, cv2.THRESH_BINARY)
            try:
                t = pytesseract.image_to_string(bw, config=cfg).strip()
                d = re.sub(r"\D", "", t)
                if d: results.append(d[0])
            except: pass
    return results


def ocr_pair(img):
    """Two-digit OCR."""
    results = []
    cfgs = [
        "--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789",
        "--oem 3 --psm 13 -c tessedit_char_whitelist=0123456789",
    ]
    for cfg in cfgs:
        try:
            t = pytesseract.image_to_string(img, config=cfg).strip()
            d = re.sub(r"\D", "", t)
            if len(d) == 2: results.append(d)
        except: pass
        for th in [120, 160]:
            _, bw = cv2.threshold(img, th, 255, cv2.THRESH_BINARY)
            try:
                t = pytesseract.image_to_string(bw, config=cfg).strip()
                d = re.sub(r"\D", "", t)
                if len(d) == 2: results.append(d)
            except: pass
    return results


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "/Users/jenineferderas/Desktop/card_image.jpg"
    img = load(path)
    h, w = img.shape[:2]
    print(f"Image: {w}x{h}")

    # Card number region
    y0, y1 = int(h*0.20), int(h*0.70)
    roi = img[y0:y1]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # ── 1. Locate '0665' with char boxes ──
    print("\n── Locating '0665' ──")
    found_0665 = None
    for cfg in ["--oem 3 --psm 6", "--oem 3 --psm 7", "--oem 3 --psm 11"]:
        for clip in [8, 16, 32]:
            c = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(4,4))
            enh = c.apply(gray)
            for src in [cv2.bitwise_not(enh), enh]:
                boxes = char_boxes(src, cfg)
                dboxes = [b for b in boxes if b["ch"].isdigit()]
                dtxt = "".join(b["ch"] for b in dboxes)
                idx = dtxt.rfind("0665")
                if idx >= 0 and found_0665 is None:
                    found_0665 = dboxes[idx:idx+4]
                    print(f"  Found! cfg={cfg}, clip={clip}")
                    print(f"  Digit text: ...{dtxt[max(0,idx-6):idx+10]}...")
                    print(f"  '0' box: x=[{found_0665[0]['x1']}, {found_0665[0]['x2']}], "
                          f"y=[{found_0665[0]['y1']}, {found_0665[0]['y2']}]")
                    break
            if found_0665: break
        if found_0665: break

    if not found_0665:
        # Fallback: manual estimate. 
        # From previous runs we know marker ≈ x=[15,1031], and suffix starts after x≈1031.
        # Typical Visa card: group4 "0665" in the rightmost ~25% of text
        # Let's estimate: 0 starts around x=1050 in ROI, digit pitch ≈ 40px
        print("  FALLBACK: Using estimated positions")
        pitch = 40
        found_0665 = [
            {"x1": 1050, "x2": 1090, "y1": 40, "y2": 100},
            {"x1": 1090, "x2": 1130, "y1": 40, "y2": 100},
            {"x1": 1130, "x2": 1170, "y1": 40, "y2": 100},
            {"x1": 1170, "x2": 1210, "y1": 40, "y2": 100},
        ]

    # ── 2. Compute digit pitch ──
    pitches = []
    for i in range(3):
        pitches.append(found_0665[i+1]["x1"] - found_0665[i]["x1"])
    pitch = sum(pitches) / len(pitches) if pitches else 40
    dh = found_0665[0]["y2"] - found_0665[0]["y1"]
    ox = found_0665[0]["x1"]  # x of '0'
    oy0 = max(0, found_0665[0]["y1"] - int(dh*0.4))
    oy1 = min(roi.shape[0], found_0665[0]["y2"] + int(dh*0.4))
    print(f"  Digit pitch: {pitch:.1f}px, digit height: {dh}px")
    print(f"  '0' of 0665 at x={ox}")

    # ── 3. Per-position analysis for 10 and 11 ──
    # Position 11 = 1 digit left of '0' (+ possible group gap)
    # Position 10 = 2 digits left of '0' (+ gap)
    print(f"\n── Deep scan positions 10 & 11 ──\n")

    for pos, label in [(11, "POS 11 (1 left of 0665)"), (10, "POS 10 (2 left of 0665)")]:
        print(f"  {label}")
        votes = Counter()
        offset_digits = 12 - pos  # pos11=1, pos10=2

        for gap_factor in [0.0, 0.3, 0.5, 0.7, 1.0, 1.3]:
            gap = pitch * gap_factor
            cx = ox - gap - pitch * (offset_digits - 0.5)
            hw = pitch * 0.6
            zx0 = max(0, int(cx - hw))
            zx1 = min(w, int(cx + hw))

            zone = gray[oy0:oy1, zx0:zx1]
            if zone.size == 0: continue

            variants = enhance(zone, scale=5)
            for v in variants:
                reads = ocr1(v)
                for r in reads:
                    votes[r] += 1

            # Also try R/G/B channels
            for ch in range(3):
                zone_ch = roi[oy0:oy1, zx0:zx1, ch]
                variants_ch = enhance(zone_ch, scale=5)
                for v in variants_ch:
                    reads = ocr1(v)
                    for r in reads:
                        votes[r] += 1

        total = sum(votes.values())
        if total:
            ranked = votes.most_common(8)
            for d, n in ranked:
                pct = n/total*100
                bar = "█" * max(1, int(pct/2))
                print(f"      '{d}': {n:4d} ({pct:5.1f}%)  {bar}")
        else:
            print("      No reads")
        print()

    # ── 4. Pair read ──
    print("  PAIR (pos 10+11 together)")
    pair_votes = Counter()
    for gap_factor in [0.0, 0.3, 0.5, 0.7, 1.0]:
        gap = pitch * gap_factor
        cx = ox - gap - pitch
        hw = pitch * 1.2
        zx0 = max(0, int(cx - hw))
        zx1 = min(w, int(cx + hw))
        zone = gray[oy0:oy1, zx0:zx1]
        if zone.size == 0: continue
        variants = enhance(zone, scale=5)
        for v in variants:
            pairs = ocr_pair(v)
            for p in pairs:
                pair_votes[p] += 1

    total = sum(pair_votes.values())
    if total:
        ranked = pair_votes.most_common(10)
        for p, n in ranked:
            pct = n/total*100
            print(f"      '{p}': {n:4d} ({pct:5.1f}%)")
    print()

    # ── 5. Transition: 3 digits before → 0665 ──
    print("  TRANSITION (last 3 hidden → 0665)")
    tvotes = Counter()
    for shift in range(-20, 21, 10):
        zx0 = max(0, int(ox - pitch*3.5 + shift))
        zx1 = min(w, found_0665[3]["x2"] + 5)
        zone = gray[oy0:oy1, zx0:zx1]
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
                    if len(d) >= 4:
                        tvotes[d] += 1
                except: pass

    total = sum(tvotes.values())
    if total:
        ranked = tvotes.most_common(15)
        for seq, n in ranked:
            pct = n/total*100
            tag = " ◄◄" if seq.endswith("0665") else (" ◄" if seq.endswith("665") else "")
            print(f"      '{seq}': {n:4d} ({pct:5.1f}%){tag}")

    # ── 6. Save debug crops ──
    print("\n── Debug images ──")
    for gap_f in [0.0, 0.5]:
        gap = pitch * gap_f
        for pos, lbl in [(11, "p11"), (10, "p10")]:
            off = 12 - pos
            cx = ox - gap - pitch*(off-0.5)
            hw = pitch*0.6
            zx0, zx1 = max(0, int(cx-hw)), min(w, int(cx+hw))
            z = gray[oy0:oy1, zx0:zx1]
            cv2.imwrite(f"/tmp/dz2_{lbl}_g{gap_f}.png", z)
            c = cv2.createCLAHE(clipLimit=32.0, tileGridSize=(3,3))
            e = cv2.bitwise_not(c.apply(z))
            big = cv2.resize(e, (e.shape[1]*8, e.shape[0]*8), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(f"/tmp/dz2_{lbl}_g{gap_f}_enh.png", big)
            print(f"  /tmp/dz2_{lbl}_g{gap_f}.png & _enh.png ({zx0}-{zx1})")


if __name__ == "__main__":
    main()
