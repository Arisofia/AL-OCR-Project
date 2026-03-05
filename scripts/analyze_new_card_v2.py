"""
Adaptive analysis of new card image - find digit positions using OCR bounding boxes
then read hidden digits.
"""
import re
from collections import Counter
from contextlib import suppress

import cv2
import numpy as np
import pytesseract
from scipy.signal import find_peaks

IMG = "/Users/jenineferderas/Downloads/20241007_002852000_iOS 3.jpg"
WL = "-c tessedit_char_whitelist=0123456789"

img = cv2.imread(IMG)
if img is None:
    raise FileNotFoundError(f"Cannot load image: {IMG}")
h, w = img.shape[:2]
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
r_ch = img[:, :, 2]

print(f"Image: {w}x{h}")

print("\n=== STEP 1: Find digit locations via OCR bounding boxes ===\n")

best_boxes = []
best_score = 0
best_info = "no successful OCR config"
best_digits = ""

for y_start_pct in range(15, 60, 5):
    for y_end_pct in range(y_start_pct + 20, min(y_start_pct + 50, 90), 5):
        y0 = int(h * y_start_pct / 100)
        y1 = int(h * y_end_pct / 100)
        roi = gray[y0:y1, :]
        roi_r = r_ch[y0:y1, :]
        rh, rw = roi.shape
        
        for src_name, src in [("gray", roi), ("rchan", roi_r)]:
            for clip in [8, 16, 32, 64]:
                enh = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(4,4)).apply(src)
                inv = cv2.bitwise_not(enh)
                scale = max(2, 600 // rh)
                up = cv2.resize(inv, (rw*scale, rh*scale), interpolation=cv2.INTER_CUBIC)
                
                for psm in [6, 7, 11]:
                    with suppress(Exception):
                        data = pytesseract.image_to_data(up, config=f"--oem 3 --psm {psm} {WL}", output_type=pytesseract.Output.DICT)
                        text_total = "".join(data["text"])
                        digits_only = re.sub(r'\D', '', text_total)
                        
                        score = len(digits_only)
                        if "4388" in digits_only: score += 20
                        if "0665" in digits_only: score += 20
                        if "438854" in digits_only: score += 30
                        
                        if score > best_score:
                            best_score = score
                            boxes = []
                            for i, txt in enumerate(data["text"]):
                                if txt.strip():
                                    bx = data["left"][i] // scale
                                    by = data["top"][i] // scale + y0
                                    bw = data["width"][i] // scale
                                    bh_val = data["height"][i] // scale
                                    boxes.append((bx, by, bw, bh_val, txt.strip()))
                            best_boxes = boxes
                            best_info = f"y=[{y0}-{y1}] {src_name} CLAHE{clip} PSM{psm}"
                            best_digits = digits_only

        for gamma in [0.3, 0.5]:
            lut = np.array([((i/255.0)**gamma)*255 for i in range(256)], np.uint8)
            bright = cv2.LUT(roi, lut)
            inv = cv2.bitwise_not(bright)
            scale = max(2, 600 // rh)
            up = cv2.resize(inv, (rw*scale, rh*scale), interpolation=cv2.INTER_CUBIC)
            for psm in [6, 7]:
                try:
                    data = pytesseract.image_to_data(up, config=f"--oem 3 --psm {psm} {WL}", output_type=pytesseract.Output.DICT)
                    text_total = "".join(data["text"])
                    digits_only = re.sub(r'\D', '', text_total)
                    score = len(digits_only)
                    if "4388" in digits_only: score += 20
                    if "0665" in digits_only: score += 20
                    if "438854" in digits_only: score += 30
                    if score > best_score:
                        best_score = score
                        boxes = []
                        for i, txt in enumerate(data["text"]):
                            if txt.strip():
                                bx = data["left"][i] // scale
                                by = data["top"][i] // scale + y0
                                bw = data["width"][i] // scale
                                bh_val = data["height"][i] // scale
                                boxes.append((bx, by, bw, bh_val, txt.strip()))
                        best_boxes = boxes
                        best_info = f"y=[{y0}-{y1}] gamma{gamma} PSM{psm}"
                        best_digits = digits_only
                except Exception:
                    pass

print(f"Best OCR result: score={best_score}")
print(f"  Config: {best_info}")
print(f"  Digits: '{best_digits}'")
print(f"  Boxes ({len(best_boxes)}):")
for bx, by, bw, bh_val, txt in best_boxes:
    print(f"    x={bx:3d}, y={by:3d}, w={bw:3d}, h={bh_val:3d}, text='{txt}'")

print("\n=== STEP 2: Heavy upscale full image OCR ===\n")

all_ocr_results = []

for scale in [4, 6, 8]:
    big = cv2.resize(gray, (w*scale, h*scale), interpolation=cv2.INTER_CUBIC)
    big_r = cv2.resize(r_ch, (w*scale, h*scale), interpolation=cv2.INTER_CUBIC)
    
    for src_name, src in [("gray", big), ("rchan", big_r)]:
        for clip in [8, 16, 32, 64]:
            enh = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(8,8)).apply(src)
            inv = cv2.bitwise_not(enh)
            for psm in [6, 11]:
                try:
                    cfg = f"--oem 3 --psm {psm} {WL}"
                    txt = pytesseract.image_to_string(inv, config=cfg).strip()
                    digits = re.sub(r'\D', '', txt)
                    score = len(digits)
                    if "4388" in digits: score += 20
                    if "0665" in digits: score += 20
                    if "438854" in digits: score += 30
                    if score >= 20:
                        all_ocr_results.append((score, digits, f"scale{scale} {src_name} CLAHE{clip} PSM{psm}"))
                except Exception:
                    pass
        
        for gamma in [0.3, 0.5]:
            lut = np.array([((i/255.0)**gamma)*255 for i in range(256)], np.uint8)
            bright = cv2.LUT(src, lut)
            inv = cv2.bitwise_not(bright)
            for psm in [6, 11]:
                try:
                    cfg = f"--oem 3 --psm {psm} {WL}"
                    txt = pytesseract.image_to_string(inv, config=cfg).strip()
                    digits = re.sub(r'\D', '', txt)
                    score = len(digits)
                    if "4388" in digits: score += 20
                    if "0665" in digits: score += 20
                    if "438854" in digits: score += 30
                    if score >= 20:
                        all_ocr_results.append((score, digits, f"scale{scale} {src_name} gamma{gamma} PSM{psm}"))
                except Exception:
                    pass

all_ocr_results.sort(key=lambda x: -x[0])
print(f"Found {len(all_ocr_results)} results with score >= 20")
for score, digits, info in all_ocr_results[:20]:
    match_4388 = "4388✓" if "4388" in digits else "      "
    match_0665 = "0665✓" if "0665" in digits else "      "
    match_full = "FULL✓" if "438854" in digits else "      "
    print(f"  score={score:3d} [{match_4388} {match_0665} {match_full}] '{digits[:40]}'  ({info})")

print("\n=== STEP 3: Edge-based digit row detection ===\n")

edges = cv2.Canny(gray, 30, 100)
row_edge_density = edges.mean(axis=1)

edge_threshold = row_edge_density.max() * 0.5
peak_rows = [y for y in range(10, h - 10) if row_edge_density[y] > edge_threshold]

if peak_rows:
    digit_y_min = min(peak_rows)
    digit_y_max = max(peak_rows)
    print(f"Edge density peak band: y=[{digit_y_min}, {digit_y_max}]")
    
    edges_r = cv2.Canny(r_ch, 30, 100)
    row_edge_r = edges_r.mean(axis=1)
    
    roi_edges = edges[digit_y_min:digit_y_max, :]
    col_density = roi_edges.mean(axis=0)
    
    peaks, props = find_peaks(col_density, height=col_density.max()*0.2, distance=10)
    print(f"Column edge peaks: {len(peaks)} found")
    for i, p in enumerate(peaks):
        print(f"  Peak {i}: x={p}, density={col_density[p]:.1f}")
    
    if len(peaks) >= 4:
        gaps = np.diff(peaks)
        print(f"\n  Gaps between peaks: {gaps}")
        
        median_gap = np.median(gaps)
        large_gaps = np.nonzero(gaps > median_gap * 1.5)[0]
        print(f"  Large gaps at indices: {large_gaps}")
else:
    print("No significant edge density peaks found")
    digit_y_min, digit_y_max = int(h*0.30), int(h*0.65)

print("\n=== STEP 4: Save enhanced images ===\n")

for name, src in [("gray", gray), ("rchan", r_ch)]:
    enh = cv2.createCLAHE(clipLimit=32.0, tileGridSize=(4,4)).apply(src)
    inv = cv2.bitwise_not(enh)
    up = cv2.resize(inv, (w*6, h*6), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(f"/tmp/new_full_{name}_enhanced.png", up)

for name, src in [("gray", gray), ("rchan", r_ch)]:
    roi = src[digit_y_min:digit_y_max, :]
    enh = cv2.createCLAHE(clipLimit=32.0, tileGridSize=(4,4)).apply(roi)
    inv = cv2.bitwise_not(enh)
    up = cv2.resize(inv, (roi.shape[1]*6, roi.shape[0]*6), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(f"/tmp/new_row_{name}_enhanced.png", up)

edges_vis = cv2.resize(edges, (w*4, h*4), interpolation=cv2.INTER_NEAREST)
cv2.imwrite("/tmp/new_edges.png", edges_vis)

print("Saved to /tmp/new_full_*_enhanced.png, /tmp/new_row_*_enhanced.png, /tmp/new_edges.png")

print("\n=== STEP 5: Column variance profile (digit position finder) ===\n")

var_window = 5
for src_name, src in [("gray", gray), ("rchan", r_ch)]:
    roi = src[digit_y_min:digit_y_max, :]
    col_var = np.zeros(roi.shape[1])
    for x in range(roi.shape[1]):
        col = roi[:, x].astype(float)
        col_var[x] = col.var()
    
    kernel = np.ones(5) / 5
    col_var_smooth = np.convolve(col_var, kernel, mode='same')
    
    try:
        peaks_v, _ = find_peaks(col_var_smooth, height=np.median(col_var_smooth)*1.5, distance=8)
        print(f"  {src_name}: {len(peaks_v)} variance peaks")
        for i, p in enumerate(peaks_v[:20]):
            print(f"    Peak {i}: x={p}, var={col_var_smooth[p]:.1f}")
    except Exception:
        pass

print("\n=== DONE ===")
