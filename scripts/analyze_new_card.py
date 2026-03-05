"""
Full analysis of the NEW card image (no grey background).
/Users/jenineferderas/Downloads/20241007_002852000_iOS 3.jpg

Steps:
1. Detect card number row (embossed digits)
2. Find digit column positions
3. Verify known digits match
4. OCR + edge + template analysis on hidden positions
5. Luhn validation
"""
import re
from collections import Counter, defaultdict
from contextlib import suppress

import cv2
import numpy as np
import pytesseract

IMG = "/Users/jenineferderas/Downloads/20241007_002852000_iOS 3.jpg"
WL = "-c tessedit_char_whitelist=0123456789"

img_bgr = cv2.imread(IMG)
if img_bgr is None:
    raise FileNotFoundError(f"Cannot load image: {IMG}")
h, w = img_bgr.shape[:2]
gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
r_ch = img_bgr[:, :, 2]

print(f"Image: {w}x{h}")
print(f"Mean brightness: {gray.mean():.0f}")
print()

print("=" * 60)
print("STEP 1: Locate digit row")
print("=" * 60)

best_roi = None
best_contrast = 0

for y_start_pct in range(20, 70, 5):
    for y_end_pct in range(y_start_pct + 15, min(y_start_pct + 40, 95), 5):
        y0 = int(h * y_start_pct / 100)
        y1 = int(h * y_end_pct / 100)
        band = gray[y0:y1, :]
        col_profile = band.mean(axis=0)
        contrast = col_profile.max() - col_profile.min()
        if contrast > best_contrast:
            best_contrast = contrast
            best_roi = (y0, y1)

if best_roi is None:
    best_roi = (int(h * 0.25), int(h * 0.55))

y0, y1 = best_roi
print(f"Best ROI: y=[{y0}, {y1}] ({y0/h*100:.0f}%-{y1/h*100:.0f}%), contrast={best_contrast:.0f}")

roi_candidates = [
    (int(h * 0.30), int(h * 0.60)),
    (int(h * 0.25), int(h * 0.55)),
    (int(h * 0.35), int(h * 0.65)),
    best_roi,
]

print("\n" + "=" * 60)
print("STEP 2: Full ROI OCR to detect digit positions")
print("=" * 60)

for roi_y0, roi_y1 in roi_candidates:
    roi = gray[roi_y0:roi_y1, :]
    roi_h, roi_w = roi.shape
    
    for clip in [8, 16, 32, 64]:
        enh = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(4, 4)).apply(roi)
        inv = cv2.bitwise_not(enh)
        
        scale = max(1, 600 // roi_h)
        up = cv2.resize(inv, (roi_w * scale, roi_h * scale), interpolation=cv2.INTER_CUBIC)
        
        for psm in [6, 7, 11, 13]:
            with suppress(Exception):
                cfg = f"--oem 3 --psm {psm} {WL}"
                txt = pytesseract.image_to_string(up, config=cfg).strip()
                digits = re.sub(r'\D', '', txt)
                if len(digits) >= 12:
                    print(f"  ROI [{roi_y0}-{roi_y1}] CLAHE{clip} PSM{psm}: '{digits}' (len={len(digits)})")
                    if '4388' in digits or '0665' in digits:
                        print("    >>> MATCH! Contains known prefix/suffix")

    roi_r = r_ch[roi_y0:roi_y1, :]
    for clip in [16, 32, 64]:
        enh = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(4, 4)).apply(roi_r)
        inv = cv2.bitwise_not(enh)
        scale = max(1, 600 // roi_h)
        up = cv2.resize(inv, (roi_w * scale, roi_h * scale), interpolation=cv2.INTER_CUBIC)
        for psm in [6, 7, 11]:
            with suppress(Exception):
                cfg = f"--oem 3 --psm {psm} {WL}"
                txt = pytesseract.image_to_string(up, config=cfg).strip()
                digits = re.sub(r'\D', '', txt)
                if len(digits) >= 12:
                    print(f"  ROI [{roi_y0}-{roi_y1}] R-ch CLAHE{clip} PSM{psm}: '{digits}' (len={len(digits)})")
                    if '4388' in digits or '0665' in digits:
                        print("    >>> MATCH!")

    for gamma in [0.3, 0.5]:
        lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], np.uint8)
        bright = cv2.LUT(roi, lut)
        inv = cv2.bitwise_not(bright)
        scale = max(1, 600 // roi_h)
        up = cv2.resize(inv, (roi_w * scale, roi_h * scale), interpolation=cv2.INTER_CUBIC)
        for psm in [6, 7]:
            with suppress(Exception):
                cfg = f"--oem 3 --psm {psm} {WL}"
                txt = pytesseract.image_to_string(up, config=cfg).strip()
                digits = re.sub(r'\D', '', txt)
                if len(digits) >= 12:
                    print(f"  ROI [{roi_y0}-{roi_y1}] Gamma{gamma} PSM{psm}: '{digits}' (len={len(digits)})")
                    if '4388' in digits or '0665' in digits:
                        print("    >>> MATCH!")


print("\n" + "=" * 60)
print("STEP 3: Column brightness analysis for digit positions")
print("=" * 60)

scale_x = w / 1568.0

ORIG_CENTERS = {
    0: 197, 1: 272, 2: 347, 3: 422,
    4: 497, 5: 572, 6: 647, 7: 722,
    8: 833, 9: 908, 10: 983, 11: 1058,
    12: 1133, 13: 1208, 14: 1283, 15: 1358,
}

NEW_CENTERS = {pos: int(cx * scale_x) for pos, cx in ORIG_CENTERS.items()}

print(f"Scale factor x: {scale_x:.3f}")
print("Estimated digit centers:")
for pos in range(16):
    known = "✓" if pos in {0,1,2,3,4,5,12,13,14,15} else "?"
    print(f"  pos{pos:2d}: x={NEW_CENTERS[pos]:4d} [{known}]")

scale_y = h / 499.0
est_y0 = int(499 * 0.25 * scale_y)
est_y1 = int(499 * 0.55 * scale_y)
print(f"\nEstimated digit ROI y: [{est_y0}, {est_y1}]")

print("\n" + "=" * 60)
print("STEP 4: Per-position digit reading (new image)")
print("=" * 60)

KNOWN_DIGITS = {0:"4", 1:"3", 2:"8", 3:"8", 4:"5", 5:"4", 12:"0", 13:"6", 14:"6", 15:"5"}
HALF_NEW = int(34 * scale_x)
SCALE_UP = 4

y_ranges = [
    (est_y0, est_y1),
    (int(h*0.30), int(h*0.60)),
    (int(h*0.25), int(h*0.55)),
    (int(h*0.35), int(h*0.65)),
    (int(h*0.20), int(h*0.50)),
]

for yi, (ry0, ry1) in enumerate(y_ranges):
    roi_g = gray[ry0:ry1, :]
    roi_r = r_ch[ry0:ry1, :]
    rh_loc, rw_loc = roi_g.shape
    
    correct = 0
    total_known = 0
    
    for pos in sorted(KNOWN_DIGITS.keys()):
        expected = KNOWN_DIGITS[pos]
        cx = NEW_CENTERS[pos]
        x0 = max(0, cx - HALF_NEW)
        x1 = min(rw_loc, cx + HALF_NEW)
        
        zone_g = roi_g[:, x0:x1]
        zone_r = roi_r[:, x0:x1]
        
        zg_up = cv2.resize(zone_g, (zone_g.shape[1]*SCALE_UP, zone_g.shape[0]*SCALE_UP), interpolation=cv2.INTER_CUBIC)
        zr_up = cv2.resize(zone_r, (zone_r.shape[1]*SCALE_UP, zone_r.shape[0]*SCALE_UP), interpolation=cv2.INTER_CUBIC)
        
        votes = Counter()
        for src in [zg_up, zr_up]:
            for clip in [8, 16, 32, 64]:
                enh = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(3,3)).apply(src)
                inv = cv2.bitwise_not(enh)
                for psm in [7, 8, 10, 13]:
                    try:
                        cfg = f"--oem 3 --psm {psm} {WL}"
                        txt = pytesseract.image_to_string(inv, config=cfg).strip()
                        d = re.sub(r'\D', '', txt)
                        if d:
                            votes[d[0]] += 1
                    except Exception:
                        pass
                    for thr in [120, 150]:
                        try:
                            _, bw = cv2.threshold(inv, thr, 255, cv2.THRESH_BINARY)
                            txt = pytesseract.image_to_string(bw, config=cfg).strip()
                            d = re.sub(r'\D', '', txt)
                            if d:
                                votes[d[0]] += 1
                        except Exception:
                            pass
            for gamma in [0.3, 0.5]:
                lut = np.array([((i/255.0)**gamma)*255 for i in range(256)], np.uint8)
                bright = cv2.bitwise_not(cv2.LUT(src, lut))
                for psm in [10, 13]:
                    try:
                        cfg = f"--oem 3 --psm {psm} {WL}"
                        txt = pytesseract.image_to_string(bright, config=cfg).strip()
                        d = re.sub(r'\D', '', txt)
                        if d:
                            votes[d[0]] += 1
                    except Exception:
                        pass

        total_known += 1
        best = votes.most_common(1)[0][0] if votes else "?"
        hit = "✓" if best == expected else "✗"
        if best == expected:
            correct += 1
    
    accuracy = correct / total_known * 100
    print(f"  Y-range [{ry0}-{ry1}]: {correct}/{total_known} known digits correct ({accuracy:.0f}%)")
    
    if accuracy < 50:
        continue
    
    print("\n  >>> Using this Y-range for hidden position analysis <<<")
    print()
    
    print("  Known digit verification:")
    for pos in sorted(KNOWN_DIGITS.keys()):
        expected = KNOWN_DIGITS[pos]
        cx = NEW_CENTERS[pos]
        x0 = max(0, cx - HALF_NEW)
        x1 = min(rw_loc, cx + HALF_NEW)
        zone_g = roi_g[:, x0:x1]
        zone_r = roi_r[:, x0:x1]
        zg_up = cv2.resize(zone_g, (zone_g.shape[1]*SCALE_UP, zone_g.shape[0]*SCALE_UP), interpolation=cv2.INTER_CUBIC)
        zr_up = cv2.resize(zone_r, (zone_r.shape[1]*SCALE_UP, zone_r.shape[0]*SCALE_UP), interpolation=cv2.INTER_CUBIC)
        
        votes = Counter()
        for src in [zg_up, zr_up]:
            for clip in [8, 16, 32, 64]:
                enh = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(3,3)).apply(src)
                inv = cv2.bitwise_not(enh)
                for psm in [7, 8, 10, 13]:
                    try:
                        txt = pytesseract.image_to_string(inv, config=f"--oem 3 --psm {psm} {WL}").strip()
                        d = re.sub(r'\D', '', txt)
                        if d: votes[d[0]] += 1
                    except Exception: pass
                    for thr in [120, 150]:
                        try:
                            _, bw = cv2.threshold(inv, thr, 255, cv2.THRESH_BINARY)
                            txt = pytesseract.image_to_string(bw, config=f"--oem 3 --psm {psm} {WL}").strip()
                            d = re.sub(r'\D', '', txt)
                            if d: votes[d[0]] += 1
                        except Exception: pass
            for gamma in [0.3, 0.5]:
                lut = np.array([((i/255.0)**gamma)*255 for i in range(256)], np.uint8)
                bright = cv2.bitwise_not(cv2.LUT(src, lut))
                for psm in [10, 13]:
                    try:
                        txt = pytesseract.image_to_string(bright, config=f"--oem 3 --psm {psm} {WL}").strip()
                        d = re.sub(r'\D', '', txt)
                        if d: votes[d[0]] += 1
                    except Exception: pass

        total = sum(votes.values())
        top3 = ", ".join(f"'{k}'={v/total:.0%}" for k, v in votes.most_common(3)) if total else "no reads"
        best = votes.most_common(1)[0][0] if votes else "?"
        hit = "✓" if best == expected else "✗"
        print(f"    pos{pos:2d} expect='{expected}' got='{best}' [{hit}]  {top3}")
    
    print("\n  HIDDEN POSITIONS (6-11):")
    hidden_results = {}
    for pos in range(6, 12):
        cx = NEW_CENTERS[pos]
        x0 = max(0, cx - HALF_NEW)
        x1 = min(rw_loc, cx + HALF_NEW)
        zone_g = roi_g[:, x0:x1]
        zone_r = roi_r[:, x0:x1]
        zg_up = cv2.resize(zone_g, (zone_g.shape[1]*SCALE_UP, zone_g.shape[0]*SCALE_UP), interpolation=cv2.INTER_CUBIC)
        zr_up = cv2.resize(zone_r, (zone_r.shape[1]*SCALE_UP, zone_r.shape[0]*SCALE_UP), interpolation=cv2.INTER_CUBIC)
        
        votes = Counter()
        for dx in [-5, -3, 0, 3, 5]:
            cx2 = NEW_CENTERS[pos] + dx
            x0d = max(0, cx2 - HALF_NEW)
            x1d = min(rw_loc, cx2 + HALF_NEW)
            zg2 = roi_g[:, x0d:x1d]
            zr2 = roi_r[:, x0d:x1d]
            zg2_up = cv2.resize(zg2, (zg2.shape[1]*SCALE_UP, zg2.shape[0]*SCALE_UP), interpolation=cv2.INTER_CUBIC)
            zr2_up = cv2.resize(zr2, (zr2.shape[1]*SCALE_UP, zr2.shape[0]*SCALE_UP), interpolation=cv2.INTER_CUBIC)
            
            for src in [zg2_up, zr2_up]:
                for clip in [8, 16, 32, 64]:
                    enh = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(3,3)).apply(src)
                    inv = cv2.bitwise_not(enh)
                    for psm in [7, 8, 10, 13]:
                        try:
                            txt = pytesseract.image_to_string(inv, config=f"--oem 3 --psm {psm} {WL}").strip()
                            d = re.sub(r'\D', '', txt)
                            if d: votes[d[0]] += 1
                        except Exception: pass
                        for thr in [100, 120, 140, 150, 170]:
                            try:
                                _, bw = cv2.threshold(inv, thr, 255, cv2.THRESH_BINARY)
                                txt = pytesseract.image_to_string(bw, config=f"--oem 3 --psm {psm} {WL}").strip()
                                d = re.sub(r'\D', '', txt)
                                if d: votes[d[0]] += 1
                            except Exception: pass
                for gamma in [0.3, 0.5]:
                    lut = np.array([((i/255.0)**gamma)*255 for i in range(256)], np.uint8)
                    bright = cv2.bitwise_not(cv2.LUT(src, lut))
                    for psm in [10, 13]:
                        try:
                            txt = pytesseract.image_to_string(bright, config=f"--oem 3 --psm {psm} {WL}").strip()
                            d = re.sub(r'\D', '', txt)
                            if d: votes[d[0]] += 1
                        except Exception: pass
                kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
                tophat = cv2.morphologyEx(src, cv2.MORPH_TOPHAT, kern)
                for psm in [10, 13]:
                    try:
                        txt = pytesseract.image_to_string(tophat, config=f"--oem 3 --psm {psm} {WL}").strip()
                        d = re.sub(r'\D', '', txt)
                        if d: votes[d[0]] += 1
                    except Exception: pass

        total = sum(votes.values())
        hidden_results[pos] = votes
        top5 = ", ".join(f"'{k}'={v/total:.0%}" for k, v in votes.most_common(5)) if total else "no reads"
        best = votes.most_common(1)[0][0] if votes else "?"
        print(f"    pos{pos:2d}: BEST='{best}'  [{top5}]  (total={total})")
    
    for pos in range(6, 12):
        cx = NEW_CENTERS[pos]
        x0 = max(0, cx - HALF_NEW)
        x1 = min(rw_loc, cx + HALF_NEW)
        zone_g = roi_g[:, x0:x1]
        zone_r = roi_r[:, x0:x1]
        zg_up = cv2.resize(zone_g, (zone_g.shape[1]*6, zone_g.shape[0]*6), interpolation=cv2.INTER_CUBIC)
        zr_up = cv2.resize(zone_r, (zone_r.shape[1]*6, zone_r.shape[0]*6), interpolation=cv2.INTER_CUBIC)
        enh_g = cv2.bitwise_not(cv2.createCLAHE(clipLimit=16.0, tileGridSize=(3,3)).apply(zg_up))
        enh_r = cv2.bitwise_not(cv2.createCLAHE(clipLimit=16.0, tileGridSize=(3,3)).apply(zr_up))
        cv2.imwrite(f"/tmp/new_pos{pos}_gray.png", enh_g)
        cv2.imwrite(f"/tmp/new_pos{pos}_rchan.png", enh_r)
    
    print("\n  LUHN VALIDATION:")
    from itertools import product as iprod
    
    top_per_pos = {}
    for pos in range(6, 12):
        v = hidden_results[pos]
        top_per_pos[pos] = "".join(d for d, _ in v.most_common(5)) if v else "0123456789"
    
    prefix = "438854"
    suffix = "0665"
    
    def luhn(pan):
        t = 0
        for i, c in enumerate(reversed(pan)):
            d = int(c)
            if i % 2 == 1:
                d *= 2
                if d > 9: d -= 9
            t += d
        return t % 10 == 0
    
    valid = []
    for combo in iprod(*[top_per_pos[p] for p in range(6, 12)]):
        mid = "".join(combo)
        pan = prefix + mid + suffix
        if luhn(pan):
            score = 1.0
            for idx, dig in enumerate(mid):
                p = 6 + idx
                v = hidden_results[p]
                total = sum(v.values())
                score *= (v.get(dig, 0) / total) if total else 0.01
            valid.append((pan, score, mid))
    
    valid.sort(key=lambda x: -x[1])
    print(f"  Combinations: {len(list(iprod(*[top_per_pos[p] for p in range(6,12)])))}")
    print(f"  Luhn-valid: {len(valid)}")
    print("\n  Top-10 PANs:")
    for i, (pan, score, mid) in enumerate(valid[:10], 1):
        formatted = f"{pan[:4]} {pan[4:8]} {pan[8:12]} {pan[12:16]}"
        print(f"    {i:3d}. {formatted}  score={score:.6e}  hidden=[{mid}]")
    
    if valid:
        best_pan = valid[0][0]
        formatted = f"{best_pan[:4]} {best_pan[4:8]} {best_pan[8:12]} {best_pan[12:16]}"
        print(f"\n  >>> BEST: {formatted}")
    
    break

print("\nDone.")
