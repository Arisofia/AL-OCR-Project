"""Rescan ALL hidden positions 6-11 with improved pipeline."""
import contextlib
import re
from collections import Counter
import cv2
import numpy as np
import pytesseract

IMG = "/Users/jenineferderas/Desktop/card_image.jpg"
WL = "-c tessedit_char_whitelist=0123456789"
OCR_TIMEOUT_SEC = 1
FAST_OCR_SWEEP = True
MAX_OCR_CALLS_PER_POS = 900 if FAST_OCR_SWEEP else 3000

img = cv2.imread(IMG)
if img is None:
    raise FileNotFoundError(f"Cannot load image: {IMG}")
h, w = img.shape[:2]
y0, y1 = int(h * 0.25), int(h * 0.55)
roi = img[y0:y1]
gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
rh, rw = gray.shape

CENTERS = {6: 647, 7: 722, 8: 833, 9: 908, 10: 983, 11: 1058}
HALF = 34
SCALE = 4
CONFIGS = [
    f"--oem 3 --psm 10 {WL}",
    f"--oem 3 --psm 13 {WL}",
]
if not FAST_OCR_SWEEP:
    CONFIGS.append(f"--oem 3 --psm 8 {WL}")
THRESHOLDS = [120, 150] if FAST_OCR_SWEEP else [100, 120, 150]


def upscale(im, factor=SCALE):
    return cv2.resize(im, (im.shape[1]*factor, im.shape[0]*factor),
                      interpolation=cv2.INTER_CUBIC)


def enhance_variants(zone):
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


def _sweep_thresholds(image_in, cfg_in, budget_left, ocr_fn):
    """Run OCR over binary-threshold variants within a call budget."""
    local_reads = []
    local_calls = 0
    for thr in THRESHOLDS:
        if local_calls >= budget_left:
            break
        _, bw = cv2.threshold(image_in, thr, 255, cv2.THRESH_BINARY)
        if digit := ocr_fn(bw, cfg_in):
            local_reads.append(digit)
        local_calls += 1
    return local_reads, local_calls


def ocr_digit(im, budget):
    def _ocr_one(src, cfg):
        with contextlib.suppress(Exception):
            txt = pytesseract.image_to_string(
                src,
                config=cfg,
                timeout=OCR_TIMEOUT_SEC,
            ).strip()
            if d := re.sub(r"\D", "", txt):
                return d[0]
        return None

    reads = []
    calls_used = 0

    if budget <= 0:
        return reads, calls_used
    for cfg in CONFIGS:
        if calls_used >= budget:
            break
        if digit := _ocr_one(im, cfg):
            reads.append(digit)
        calls_used += 1
        threshold_reads, threshold_calls = _sweep_thresholds(
            im,
            cfg,
            budget - calls_used,
            _ocr_one,
        )
        reads.extend(threshold_reads)
        calls_used += threshold_calls
    return reads, calls_used


print(f"Image: {w}x{h}, ROI: {rw}x{rh}")
print("Scanning positions 6-11 with improved pipeline\n")

all_results = {}

for pos, cx in sorted(CENTERS.items()):
    votes = Counter()
    budget_remaining = MAX_OCR_CALLS_PER_POS
    dx_offsets = [-5, -2, 0, 2, 5] if FAST_OCR_SWEEP else [-10, -5, -2, 0, 2, 5, 10]

    for dx in dx_offsets:
        if budget_remaining <= 0:
            break
        x0 = max(0, cx + dx - HALF)
        x1 = min(rw, cx + dx + HALF)
        zone = gray[:, x0:x1]
        zone_up = upscale(zone)

        gray_enh = enhance_variants(zone_up)
        if FAST_OCR_SWEEP:
            gray_enh = gray_enh[:6]
        for enh in gray_enh:
            reads, calls_used = ocr_digit(enh, budget_remaining)
            budget_remaining -= calls_used
            for d in reads:
                votes[d] += 1
            if budget_remaining <= 0:
                break

        if budget_remaining <= 0:
            break
        zone_r = roi[:, x0:x1, 2]
        zone_r_up = upscale(zone_r)
        for enh in enhance_variants(zone_r_up)[:4]:
            reads, calls_used = ocr_digit(enh, budget_remaining)
            budget_remaining -= calls_used
            for d in reads:
                votes[d] += 1
            if budget_remaining <= 0:
                break

    total = sum(votes.values())
    all_results[pos] = (votes, total)
    calls_done = MAX_OCR_CALLS_PER_POS - budget_remaining

    print(f"POS {pos} (cx={cx}): {total} reads  | OCR calls {calls_done}/{MAX_OCR_CALLS_PER_POS}")
    if budget_remaining <= 0:
        print("  NOTE: OCR call budget reached; using accumulated votes.")
    for d, n in votes.most_common(5):
        pct = n / total * 100 if total else 0
        bar = "#" * max(1, int(pct / 2))
        print(f"  '{d}': {n:4d} ({pct:5.1f}%)  {bar}")
    if votes:
        best = votes.most_common(1)[0]
        print(f"  >>> '{best[0]}' ({best[1]/total:.0%})")
    print()

print("=" * 50)
print("SUMMARY — Updated evidence for all hidden positions")
print("=" * 50)
print(f"{'Pos':>3}  {'#1':>8}  {'#2':>8}  {'#3':>8}  {'Total':>5}")
print("-" * 50)
for pos in range(6, 12):
    votes, total = all_results[pos]
    mc = votes.most_common(3)
    cols = [f"'{d}'={n/total:.0%}" for d, n in mc]
    while len(cols) < 3:
        cols.append("")
    print(f"  {pos}  {cols[0]:>8}  {cols[1]:>8}  {cols[2]:>8}  {total:>5}")

print("\n# Python dict for WEIGHTS update:")
print("WEIGHTS = {")
for pos in range(6, 12):
    votes, total = all_results[pos]
    items = [f'"{d}": {n/total:.2f}' for d, n in votes.most_common(5)]
    print(f"    {pos}: {{{', '.join(items)}}},")
print("}")

print("\n# TOP_PER_POS update:")
print("TOP_PER_POS = {")
for pos in range(6, 12):
    votes, total = all_results[pos]
    tops = "".join(d for d, _ in votes.most_common(4))
    print(f'    {pos}: "{tops}",')
print("}")
