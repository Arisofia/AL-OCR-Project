"""
Read the full number line from new card image using mapped positions.
Also use edge-based structural analysis on hidden positions.
"""
import re
from collections import Counter

import cv2
import numpy as np
import pytesseract

IMG2 = "/Users/jenineferderas/Downloads/20241007_002852000_iOS 3.jpg"
WL = "-c tessedit_char_whitelist=0123456789"

img2 = cv2.imread(IMG2)
h2, w2 = img2.shape[:2]
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
r2 = img2[:, :, 2]


print(f"Image: {w2}x{h2}\n")

POS_X = {
    0:45, 1:66, 2:88, 3:110, 4:131, 5:153,
    6:175, 7:197, 8:229, 9:251, 10:273, 11:294,
    12:316, 13:338, 14:360, 15:382
}
POS_Y = 162

print("=" * 60)
print("PART 1: Full number line OCR")
print("=" * 60)

x_left = POS_X[0] - 15
x_right = POS_X[15] + 15

all_line_reads = []

for y_offset in range(-20, 25, 3):
    cy = POS_Y + y_offset
    for half_h in [15, 20, 25, 30]:
        y0 = max(0, cy - half_h)
        y1 = min(h2, cy + half_h)
        
        strip_g = gray2[y0:y1, x_left:x_right]
        strip_r = r2[y0:y1, x_left:x_right]
        
        for ch_name, ch in [("g", strip_g), ("r", strip_r)]:
            for scale in [4, 6, 8]:
                up = cv2.resize(ch, (ch.shape[1]*scale, ch.shape[0]*scale), interpolation=cv2.INTER_CUBIC)
                
                for clip in [8, 16, 32, 64]:
                    enh = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(4,4)).apply(up)
                    inv = cv2.bitwise_not(enh)
                    
                    for psm in [6, 7]:
                        try:
                            txt = pytesseract.image_to_string(inv, config=f"--oem 3 --psm {psm} {WL}").strip()
                            digits = re.sub(r'\D', '', txt)
                            if len(digits) >= 12:
                                has_4388 = "4388" in digits
                                has_0665 = "0665" in digits
                                has_438854 = "438854" in digits
                                score = len(digits)
                                if has_4388: score += 20
                                if has_0665: score += 20
                                if has_438854: score += 30
                                all_line_reads.append((score, digits, f"y{y_offset:+d}_h{half_h}_{ch_name}_s{scale}_C{clip}_P{psm}"))
                        except Exception:
                            pass
                    
                    for thr in [120, 140, 160]:
                        _, bw = cv2.threshold(inv, thr, 255, cv2.THRESH_BINARY)
                        for psm in [6, 7]:
                            try:
                                txt = pytesseract.image_to_string(bw, config=f"--oem 3 --psm {psm} {WL}").strip()
                                digits = re.sub(r'\D', '', txt)
                                if len(digits) >= 12:
                                    score = len(digits)
                                    if "4388" in digits: score += 20
                                    if "0665" in digits: score += 20
                                    if "438854" in digits: score += 30
                                    all_line_reads.append((score, digits, f"y{y_offset:+d}_h{half_h}_{ch_name}_s{scale}_C{clip}_T{thr}_P{psm}"))
                            except Exception:
                                pass
                
                for gamma in [0.3, 0.5]:
                    lut = np.array([((i/255.0)**gamma)*255 for i in range(256)], np.uint8)
                    bright = cv2.LUT(up, lut)
                    inv = cv2.bitwise_not(bright)
                    for psm in [6, 7]:
                        try:
                            txt = pytesseract.image_to_string(inv, config=f"--oem 3 --psm {psm} {WL}").strip()
                            digits = re.sub(r'\D', '', txt)
                            if len(digits) >= 12:
                                score = len(digits)
                                if "4388" in digits: score += 20
                                if "0665" in digits: score += 20
                                if "438854" in digits: score += 30
                                all_line_reads.append((score, digits, f"y{y_offset:+d}_h{half_h}_{ch_name}_s{scale}_g{gamma}_P{psm}"))
                        except Exception:
                            pass

all_line_reads.sort(key=lambda x: -x[0])
print(f"\nTotal line reads >= 12 digits: {len(all_line_reads)}")
print("\nTop-30 reads:")
for score, digits, info in all_line_reads[:30]:
    marks = ""
    if "438854" in digits: marks += " FULL_PREFIX"
    elif "4388" in digits: marks += " HAS_4388"
    if "0665" in digits: marks += " HAS_0665"
    print(f"  [{score:3d}] '{digits}' ({info}){marks}")

print("\n--- Extracted PANs ---")
pan_votes = Counter()
for score, digits, info in all_line_reads:
    idx = digits.find("438854")
    if idx >= 0 and len(digits) >= idx + 16:
        pan = digits[idx:idx+16]
        pan_votes[pan] += 1
    idx = digits.find("0665")
    if idx >= 0 and idx + 4 == len(digits) or (idx >= 0 and idx + 4 <= len(digits)):
        end = idx + 4
        start = end - 16
        if start >= 0:
            pan = digits[start:end]
            if pan.startswith("4388"):
                pan_votes[pan] += 1

print(f"\nPANs found (unique): {len(pan_votes)}")
for pan, count in pan_votes.most_common(20):
    formatted = f"{pan[:4]} {pan[4:8]} {pan[8:12]} {pan[12:16]}"
    
    t = 0
    for i, c in enumerate(reversed(pan)):
        d = int(c)
        if i % 2 == 1: d *= 2; d = d - 9 if d > 9 else d
        t += d
    valid = "✓LUHN" if t % 10 == 0 else "     "
    
    print(f"  {formatted}  x{count}  {valid}")

print(f"\n{'='*60}")
print("PART 2: Edge-based structural analysis (positions 8-10)")
print(f"{'='*60}")

HALF = 12

def analyze_structure(zone):
    """Classify digit structure based on edge patterns."""
    if zone.size == 0:
        return {}
    
    h_z, w_z = zone.shape
    
    edges = cv2.Canny(zone, 30, 100)
    
    mid_y = h_z // 2
    mid_x = w_z // 2
    
    tl = edges[:mid_y, :mid_x].sum()
    tr = edges[:mid_y, mid_x:].sum()
    bl = edges[mid_y:, :mid_x].sum()
    br = edges[mid_y:, mid_x:].sum()
    total = tl + tr + bl + br
    
    if total == 0:
        return {"total": 0}
    
    features = {
        "total": total,
        "top": (tl + tr) / total,
        "bottom": (bl + br) / total,
        "left": (tl + bl) / total,
        "right": (tr + br) / total,
        "tl": tl / total,
        "tr": tr / total,
        "bl": bl / total,
        "br": br / total,
    }
    
    sobelx = cv2.Sobel(zone, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(zone, cv2.CV_64F, 0, 1, ksize=3)
    h_edges = np.abs(sobely).sum()
    v_edges = np.abs(sobelx).sum()
    features["h_v_ratio"] = h_edges / (v_edges + 1)
    
    return features

DIGIT_PROFILES = {
    '0': {"top": 0.50, "bottom": 0.50, "left": 0.50, "right": 0.50, "h_v": 0.8},
    '1': {"top": 0.40, "bottom": 0.60, "left": 0.30, "right": 0.70, "h_v": 0.5},
    '2': {"top": 0.55, "bottom": 0.45, "left": 0.45, "right": 0.55, "h_v": 1.2},
    '3': {"top": 0.45, "bottom": 0.55, "left": 0.35, "right": 0.65, "h_v": 1.1},
    '4': {"top": 0.55, "bottom": 0.45, "left": 0.55, "right": 0.45, "h_v": 0.7},
    '5': {"top": 0.55, "bottom": 0.45, "left": 0.55, "right": 0.45, "h_v": 1.1},
    '6': {"top": 0.40, "bottom": 0.60, "left": 0.55, "right": 0.45, "h_v": 0.9},
    '7': {"top": 0.60, "bottom": 0.40, "left": 0.40, "right": 0.60, "h_v": 0.7},
    '8': {"top": 0.50, "bottom": 0.50, "left": 0.50, "right": 0.50, "h_v": 1.0},
    '9': {"top": 0.60, "bottom": 0.40, "left": 0.45, "right": 0.55, "h_v": 0.9},
}

def match_digit(features):
    """Match features against digit profiles. Return sorted scores."""
    if features.get("total", 0) == 0:
        return []
    
    scores = {}
    for d, prof in DIGIT_PROFILES.items():
        dist = 0
        dist += (features["top"] - prof["top"]) ** 2
        dist += (features["bottom"] - prof["bottom"]) ** 2
        dist += (features["left"] - prof["left"]) ** 2
        dist += (features["right"] - prof["right"]) ** 2
        dist += (features["h_v_ratio"] / 2 - prof["h_v"] / 2) ** 2
        scores[d] = dist
    
    return sorted(scores.items(), key=lambda x: x[1])

print("\nValidating on known digits:")
KNOWN = {0:"4", 1:"3", 2:"8", 3:"8", 4:"5", 5:"4", 12:"0", 13:"6", 14:"6", 15:"5"}
correct = 0
for pos in sorted(KNOWN.keys()):
    expected = KNOWN[pos]
    mx = POS_X[pos]
    
    best_over_offsets = Counter()
    for y_off in range(-10, 15, 3):
        cy = POS_Y + y_off
        y0 = max(0, cy - 18)
        y1 = min(h2, cy + 18)
        x0 = max(0, mx - HALF)
        x1 = min(w2, mx + HALF)
        
        for ch in [gray2, r2]:
            zone = ch[y0:y1, x0:x1]
            up = cv2.resize(zone, (zone.shape[1]*4, zone.shape[0]*4), interpolation=cv2.INTER_CUBIC)
            
            for clip in [8, 16, 32]:
                enh = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(3,3)).apply(up)
                feats = analyze_structure(enh)
                matches = match_digit(feats)
                if matches:
                    best_over_offsets[matches[0][0]] += 2
                    best_over_offsets[matches[1][0]] += 1
    
    top3 = best_over_offsets.most_common(3)
    best = top3[0][0] if top3 else "?"
    hit = "✓" if best == expected else "✗"
    if best == expected: correct += 1
    t3str = ", ".join(f"'{k}'={v:.0f}" for k,v in top3)
    print(f"  pos{pos:2d}: expect='{expected}' got='{best}' [{hit}]  ({t3str})")

print(f"\nKnown accuracy: {correct}/{len(KNOWN)} ({correct/len(KNOWN)*100:.0f}%)")

print("\nHidden positions (6-11) edge analysis:")
hidden_edge = {}
for pos in range(6, 12):
    mx = POS_X[pos]
    
    votes = Counter()
    for y_off in range(-10, 15, 3):
        for dx in range(-3, 4, 2):
            cy = POS_Y + y_off
            y0 = max(0, cy - 18)
            y1 = min(h2, cy + 18)
            x0 = max(0, mx + dx - HALF)
            x1 = min(w2, mx + dx + HALF)
            
            for ch in [gray2, r2]:
                zone = ch[y0:y1, x0:x1]
                up = cv2.resize(zone, (zone.shape[1]*4, zone.shape[0]*4), interpolation=cv2.INTER_CUBIC)
                
                for clip in [8, 16, 32]:
                    enh = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(3,3)).apply(up)
                    feats = analyze_structure(enh)
                    matches = match_digit(feats)
                    if matches:
                        votes[matches[0][0]] += 2
                        votes[matches[1][0]] += 1
    
    hidden_edge[pos] = votes
    top5 = ", ".join(f"'{k}'={v:.0f}" for k,v in votes.most_common(5))
    best = votes.most_common(1)[0][0] if votes else "?"
    print(f"  pos{pos:2d}: BEST='{best}'  ({top5})")

print(f"\n{'='*60}")
print("PART 3: Combined evidence + Luhn for 4388 5454 ???5 0665")
print(f"{'='*60}")

orig_evidence = {
    8: {'3': 30, '8': 20, '4': 15, '7': 25, '5': 10},
    9: {'3': 30, '8': 20, '4': 15, '7': 25, '2': 10},
    10: {'3': 30, '4': 20, '8': 15, '7': 20, '1': 15},
}

combined = {}
for pos in [8, 9, 10]:
    combined[pos] = Counter()
    for d, w in orig_evidence[pos].items():
        combined[pos][d] += w * 2
    for d, w in hidden_edge.get(pos, Counter()).items():
        combined[pos][d] += w
    
    total = sum(combined[pos].values())
    top5 = ", ".join(f"'{k}'={v/total:.0%}" for k,v in combined[pos].most_common(5))
    print(f"  pos{pos}: {top5}")

print("\nFinal Luhn-valid ranking for 4388 5454 ???5 0665:")

def luhn(pan):
    t = 0
    for i, c in enumerate(reversed(pan)):
        d = int(c)
        if i % 2 == 1: d *= 2; d = d - 9 if d > 9 else d
        t += d
    return t % 10 == 0

candidates = []
for d8 in range(10):
    for d9 in range(10):
        for d10 in range(10):
            pan = f"43885454{d8}{d9}{d10}50665"
            if luhn(pan):
                t8 = sum(combined[8].values()) or 1
                t9 = sum(combined[9].values()) or 1
                t10 = sum(combined[10].values()) or 1
                s = (combined[8].get(str(d8), 0.1)/t8) * (combined[9].get(str(d9), 0.1)/t9) * (combined[10].get(str(d10), 0.1)/t10)
                candidates.append((pan, s, str(d8)+str(d9)+str(d10)))

candidates.sort(key=lambda x: -x[1])
print("\nTop-20:")
for i, (pan, s, mid) in enumerate(candidates[:20], 1):
    formatted = f"{pan[:4]} {pan[4:8]} {pan[8:12]} {pan[12:16]}"
    print(f"  {i:3d}. {formatted}  score={s:.6e}  [{mid}]")

print(f"\n>>> BEST: {candidates[0][0][:4]} {candidates[0][0][4:8]} {candidates[0][0][8:12]} {candidates[0][0][12:16]}")
print("Done.")
