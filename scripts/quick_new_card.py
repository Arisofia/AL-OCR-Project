"""
Quick focused analysis of new card image.
We ONLY need positions 8, 9, 10 (pattern: 4388 5454 ???5 0665).
"""
import re
from collections import Counter

import cv2
import numpy as np
import pytesseract

IMG = "/Users/jenineferderas/Downloads/20241007_002852000_iOS 3.jpg"
WL = "-c tessedit_char_whitelist=0123456789"

img = cv2.imread(IMG)
h, w = img.shape[:2]
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
r_ch = img[:, :, 2]
b_ch = img[:, :, 0]
g_ch = img[:, :, 1]

print(f"Image: {w}x{h}\n")


all_reads = []
configs_tried = 0

for scale in [3, 4, 5, 6, 8]:
    for ch_name, ch in [("gray", gray), ("rchan", r_ch), ("blue", b_ch), ("green", g_ch)]:
        big = cv2.resize(ch, (w*scale, h*scale), interpolation=cv2.INTER_CUBIC)
        
        for clip in [4, 8, 16, 32, 64]:
            enh = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(8,8)).apply(big)
            inv = cv2.bitwise_not(enh)
            
            for psm in [6, 7, 11]:
                configs_tried += 1
                try:
                    txt = pytesseract.image_to_string(inv, config=f"--oem 3 --psm {psm} {WL}").strip()
                    digits = re.sub(r'\D', '', txt)
                    if len(digits) >= 14:
                        all_reads.append((digits, f"s{scale}_{ch_name}_C{clip}_P{psm}"))
                        if "4388" in digits or "0665" in digits:
                            print(f"  HIT: '{digits}' ({f's{scale}_{ch_name}_C{clip}_P{psm}'})")
                except Exception:
                    pass
            
            for thr in [100, 120, 140, 160]:
                _, bw = cv2.threshold(inv, thr, 255, cv2.THRESH_BINARY)
                for psm in [6, 7]:
                    configs_tried += 1
                    try:
                        txt = pytesseract.image_to_string(bw, config=f"--oem 3 --psm {psm} {WL}").strip()
                        digits = re.sub(r'\D', '', txt)
                        if len(digits) >= 14:
                            all_reads.append((digits, f"s{scale}_{ch_name}_C{clip}_T{thr}_P{psm}"))
                            if "4388" in digits or "0665" in digits:
                                print(f"  HIT: '{digits}' ({f's{scale}_{ch_name}_C{clip}_T{thr}_P{psm}'})")
                    except Exception:
                        pass
        
        for gamma in [0.3, 0.5, 2.0, 3.0]:
            lut = np.array([((i/255.0)**gamma)*255 for i in range(256)], np.uint8)
            bright = cv2.LUT(big, lut)
            inv = cv2.bitwise_not(bright)
            for psm in [6, 7, 11]:
                configs_tried += 1
                try:
                    txt = pytesseract.image_to_string(inv, config=f"--oem 3 --psm {psm} {WL}").strip()
                    digits = re.sub(r'\D', '', txt)
                    if len(digits) >= 14:
                        all_reads.append((digits, f"s{scale}_{ch_name}_g{gamma}_P{psm}"))
                        if "4388" in digits or "0665" in digits:
                            print(f"  HIT: '{digits}' ({f's{scale}_{ch_name}_g{gamma}_P{psm}'})")
                except Exception:
                    pass

for y_pct_start in [25, 30, 35, 40]:
    for y_pct_end in [55, 60, 65, 70]:
        y0 = int(h * y_pct_start / 100)
        y1 = int(h * y_pct_end / 100)
        roi_g = gray[y0:y1, :]
        roi_r = r_ch[y0:y1, :]
        rh = y1 - y0
        
        for scale in [4, 6, 8, 10]:
            for ch_name, ch in [("gray_roi", roi_g), ("rchan_roi", roi_r)]:
                big = cv2.resize(ch, (w*scale, rh*scale), interpolation=cv2.INTER_CUBIC)
                
                for clip in [8, 16, 32, 64]:
                    enh = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(4,4)).apply(big)
                    inv = cv2.bitwise_not(enh)
                    
                    for psm in [6, 7]:
                        configs_tried += 1
                        try:
                            txt = pytesseract.image_to_string(inv, config=f"--oem 3 --psm {psm} {WL}").strip()
                            digits = re.sub(r'\D', '', txt)
                            if len(digits) >= 14:
                                all_reads.append((digits, f"ROI{y_pct_start}-{y_pct_end}_s{scale}_{ch_name}_C{clip}_P{psm}"))
                                if "4388" in digits or "0665" in digits:
                                    print(f"  HIT: '{digits}' ({f'ROI{y_pct_start}-{y_pct_end}_s{scale}_{ch_name}_C{clip}_P{psm}'})")
                        except Exception:
                            pass

print(f"\nTotal configs tried: {configs_tried}")
print(f"Total reads >= 14 digits: {len(all_reads)}")

matched = []
for digits, info in all_reads:
    if "4388" in digits or "0665" in digits:
        matched.append((digits, info))

print(f"\nReads containing '4388' or '0665': {len(matched)}")
for digits, info in matched[:30]:
    idx = digits.find("4388")
    if idx >= 0 and len(digits) >= idx + 16:
        pan16 = digits[idx:idx+16]
        print(f"  PAN: {pan16[:4]} {pan16[4:8]} {pan16[8:12]} {pan16[12:16]}  from '{digits}'  ({info})")
    elif "0665" in digits:
        idx = digits.find("0665")
        if idx >= 4 and idx + 4 <= len(digits):
            start = max(0, idx + 4 - 16)
            pan16 = digits[start:idx+4]
            if len(pan16) == 16:
                print(f"  PAN: {pan16[:4]} {pan16[4:8]} {pan16[8:12]} {pan16[12:16]}  from '{digits}'  ({info})")
            else:
                print(f"  ??? '{digits}' ({info})")
        else:
            print(f"  ??? '{digits}' ({info})")
    else:
        print(f"  ??? '{digits}' ({info})")

print("\n=== POSITION 8, 9, 10 EVIDENCE ===\n")

pos8_votes = Counter()
pos9_votes = Counter()
pos10_votes = Counter()

for digits, info in matched:
    idx = digits.find("4388")
    if idx >= 0 and len(digits) >= idx + 16:
        pan16 = digits[idx:idx+16]
        checks = 0
        if pan16[4:6] == "54": checks += 1
        if pan16[12:16] == "0665": checks += 1
        
        if checks >= 1:
            pos8_votes[pan16[8]] += 1
            pos9_votes[pan16[9]] += 1
            pos10_votes[pan16[10]] += 1

print(f"Position 8: {pos8_votes.most_common()}")
print(f"Position 9: {pos9_votes.most_common()}")
print(f"Position 10: {pos10_votes.most_common()}")

print("\n=== PATTERN MATCHING: 438854??????0665 ===\n")

for digits, info in all_reads:
    idx_start = digits.find("438854")
    if idx_start >= 0:
        remaining = digits[idx_start:]
        if len(remaining) >= 16:
            pan16 = remaining[:16]
            if pan16.endswith("0665") or "0665" in remaining[10:20]:
                print(f"  FULL: {pan16[:4]} {pan16[4:8]} {pan16[8:12]} {pan16[12:16]}  ({info})")

print("\n=== SUFFIX PATTERN: ???50665 ===\n")
for digits, info in all_reads:
    idx = digits.find("50665")
    if idx >= 0 and idx >= 3:
        trio = digits[idx-3:idx]
        print(f"  ...{trio}50665  from '{digits}'  ({info})")

print("\nDone.")
