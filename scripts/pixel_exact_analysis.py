#!/usr/bin/env python3
"""
Exact pixel observation for ALL hidden positions (6-11).

NO OCR — pure pixel analysis only. Fast and deterministic.
For each position:
  1. Extract zone at 6x, apply best enhancements
  2. Binarize and measure: stroke presence per quadrant (TL/TR/BL/BR)
  3. Right-side gap detection (consecutive empty columns)
  4. Left-side gap detection
  5. Vertical midline crossing count (how many times the stroke crosses the center)
  6. Horizontal midline crossing count
  7. Compare these features against known digits from the card
  8. Infer best digit match by feature distance

Also: save individual enhanced images for each hidden position.
"""
from __future__ import annotations
from collections import defaultdict

import cv2
import numpy as np

IMG = "/Users/jenineferderas/Desktop/card_image.jpg"

img_bgr = cv2.imread(IMG)
h, w = img_bgr.shape[:2]
y0, y1 = int(h * 0.25), int(h * 0.55)
roi_bgr = img_bgr[y0:y1]
roi_gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
rh, rw = roi_gray.shape

CENTERS = {
    0: 197,  1: 272,  2: 347,  3: 422,
    4: 497,  5: 572,  6: 647,  7: 722,
    8: 833,  9: 908,  10: 983, 11: 1058,
    12: 1133, 13: 1208, 14: 1283, 15: 1358,
}
KNOWN = {0:"4", 1:"3", 2:"8", 3:"8", 4:"5", 5:"4", 12:"0", 13:"6", 14:"6", 15:"5"}
HALF = 34
SCALE = 6


def upscale(im):
    return cv2.resize(im, (im.shape[1]*SCALE, im.shape[0]*SCALE), interpolation=cv2.INTER_CUBIC)


def get_zone(pos, dx=0):
    cx = CENTERS[pos] + dx
    x0 = max(0, cx - HALF)
    x1 = min(rw, cx + HALF)
    return roi_gray[:, x0:x1]


def best_enhance(zone_up):
    """CLAHE 16 inverted — best for embossed silver on dark."""
    return cv2.bitwise_not(cv2.createCLAHE(clipLimit=16.0, tileGridSize=(3,3)).apply(zone_up))


def binarize(enh):
    """Otsu binarization."""
    _, b = cv2.threshold(enh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return b


def features(binary):
    """Compute structural features from a binarized digit image."""
    zh, zw = binary.shape
    _, _ = zh // 2, zw // 2
    # Trim 10% margins to avoid border noise
    margin_x = int(zw * 0.10)
    margin_y = int(zh * 0.10)
    crop = binary[margin_y:zh-margin_y, margin_x:zw-margin_x]
    ch, cw = crop.shape
    cmh, cmw = ch // 2, cw // 2

    # Quadrant densities (% of white pixels)
    tl = crop[:cmh, :cmw].astype(float).mean()
    tr = crop[:cmh, cmw:].astype(float).mean()
    bl = crop[cmh:, :cmw].astype(float).mean()
    br = crop[cmh:, cmw:].astype(float).mean()

    # Third-based densities (top/mid/bot x left/right)
    th3 = ch // 3
    t_left = crop[:th3, :cmw].astype(float).mean()
    t_right = crop[:th3, cmw:].astype(float).mean()
    m_left = crop[th3:2*th3, :cmw].astype(float).mean()
    m_right = crop[th3:2*th3, cmw:].astype(float).mean()
    b_left = crop[2*th3:, :cmw].astype(float).mean()
    b_right = crop[2*th3:, cmw:].astype(float).mean()

    # Right-half gap: max run of consecutive columns with <15% white
    right_half = crop[:, cmw:]
    col_dens = right_half.astype(float).mean(axis=0) / 255.0
    max_gap_r = _max_run_below(col_dens, 0.15)

    # Left-half gap
    left_half = crop[:, :cmw]
    col_dens_l = left_half.astype(float).mean(axis=0) / 255.0
    max_gap_l = _max_run_below(col_dens_l, 0.15)

    # Right-side: row-by-row presence
    right_row_presence = (right_half.astype(float).mean(axis=1) > 25.5).astype(int)
    right_coverage = right_row_presence.mean()  # fraction of rows with stuff on right

    # Horizontal midline crossings (transitions white↔black along middle row band)
    mid_band = crop[cmh-3:cmh+3, :].mean(axis=0)
    h_cross = _count_crossings(mid_band, 128)

    # Vertical midline crossings
    vmid_band = crop[:, cmw-3:cmw+3].mean(axis=1)
    v_cross = _count_crossings(vmid_band, 128)

    # Overall density
    total_dens = crop.astype(float).mean()

    return {
        "TL": tl, "TR": tr, "BL": bl, "BR": br,
        "t_left": t_left, "t_right": t_right,
        "m_left": m_left, "m_right": m_right,
        "b_left": b_left, "b_right": b_right,
        "gap_r": max_gap_r, "gap_l": max_gap_l,
        "right_coverage": right_coverage,
        "h_cross": h_cross, "v_cross": v_cross,
        "density": total_dens,
    }


def _max_run_below(arr, threshold):
    """Max consecutive run of values below threshold."""
    max_run = 0
    run = 0
    for v in arr:
        if v < threshold:
            run += 1
            max_run = max(max_run, run)
        else:
            run = 0
    return max_run


def _count_crossings(profile, threshold):
    """Count transitions across threshold."""
    above = profile > threshold
    return int(np.sum(np.abs(np.diff(above.astype(int)))))


# =============================================================================
# Build feature database from known digits
# =============================================================================
print(f"Image: {w}x{h}, ROI: {rw}x{rh}")
print(f"Scale: {SCALE}x\n")

print("=" * 70)
print("KNOWN DIGIT FEATURES (ground truth)")
print("=" * 70)

# Store per-digit (average across occurrences)
digit_features: dict[str, list[dict]] = defaultdict(list)

header = (f"{'Pos':>3} {'D':>1}  {'TL':>5} {'TR':>5} {'BL':>5} {'BR':>5} "
          f"{'gapR':>4} {'gapL':>4} {'Rcov':>5} {'Hx':>2} {'Vx':>2} {'dens':>5}")
print(header)
print("-" * 70)

for pos in sorted(KNOWN.keys()):
    digit = KNOWN[pos]
    zone = get_zone(pos)
    zone_up = upscale(zone)
    enh = best_enhance(zone_up)
    bn = binarize(enh)
    f = features(bn)
    digit_features[digit].append(f)

    print(f"  {pos:>2} {digit:>1}  {f['TL']:5.0f} {f['TR']:5.0f} {f['BL']:5.0f} {f['BR']:5.0f} "
          f"{f['gap_r']:>4} {f['gap_l']:>4} {f['right_coverage']:5.2f} "
          f"{f['h_cross']:>2} {f['v_cross']:>2} {f['density']:5.0f}")

# Compute average features per digit
avg_features: dict[str, dict] = {}
for digit, flist in digit_features.items():
    avg = {}
    for key in flist[0]:
        avg[key] = np.mean([f[key] for f in flist])
    avg_features[digit] = avg

print(f"\n{'Digit averages':}")
print(header.replace("Pos", "   "))
print("-" * 70)
for digit in sorted(avg_features.keys()):
    f = avg_features[digit]
    print(f"    {digit:>1}  {f['TL']:5.0f} {f['TR']:5.0f} {f['BL']:5.0f} {f['BR']:5.0f} "
          f"{f['gap_r']:>4.0f} {f['gap_l']:>4.0f} {f['right_coverage']:5.2f} "
          f"{f['h_cross']:>2.0f} {f['v_cross']:>2.0f} {f['density']:5.0f}")


# =============================================================================
# Analyse each hidden position
# =============================================================================
print("\n" + "=" * 70)
print("HIDDEN POSITION ANALYSIS")
print("=" * 70)

for pos in range(6, 12):
    zone = get_zone(pos)
    zone_up = upscale(zone)
    enh = best_enhance(zone_up)
    bn = binarize(enh)
    f = features(bn)

    print(f"\n{'─' * 70}")
    print(f"POSITION {pos} (cx={CENTERS[pos]})")
    print(f"{'─' * 70}")
    print(f"  TL={f['TL']:.0f}  TR={f['TR']:.0f}  BL={f['BL']:.0f}  BR={f['BR']:.0f}")
    print(f"  gap_R={f['gap_r']}  gap_L={f['gap_l']}  right_coverage={f['right_coverage']:.2f}")
    print(f"  h_cross={f['h_cross']}  v_cross={f['v_cross']}  density={f['density']:.0f}")

    # 6-zone detail
    print(f"  6-zone: tL={f['t_left']:.0f} tR={f['t_right']:.0f} "
          f"mL={f['m_left']:.0f} mR={f['m_right']:.0f} "
          f"bL={f['b_left']:.0f} bR={f['b_right']:.0f}")

    # Feature distance to each known digit
    print("\n  Distance to known digits (lower = more similar):")
    distances = {}
    for digit, avg_f in sorted(avg_features.items()):
        # Weighted L1 distance across structural features
        keys_weights = [
            ("TL", 1), ("TR", 2), ("BL", 1), ("BR", 2),  # right side matters more
            ("gap_r", 15), ("gap_l", 10),
            ("right_coverage", 80),
            ("h_cross", 20), ("v_cross", 15),
            ("t_right", 1.5), ("m_right", 1.5), ("b_right", 1.5),
        ]
        dist = 0
        for key, weight in keys_weights:
            dist += weight * abs(f[key] - avg_f[key])
        distances[digit] = dist

    ranked = sorted(distances.items(), key=lambda x: x[1])
    for i, (d, dist) in enumerate(ranked):
        marker = " <<<" if i == 0 else ""
        print(f"    '{d}': {dist:8.1f}{marker}")

    # Save debug images
    cv2.imwrite(f"/tmp/pos{pos}_6x_enhanced.png", enh)
    cv2.imwrite(f"/tmp/pos{pos}_6x_binary.png", bn)

    # Also save with multiple enhancements for visual inspection
    variants = []
    for clip in [4, 8, 16, 32, 64]:
        c = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(3,3))
        variants.append(cv2.bitwise_not(c.apply(zone_up)))
    for gamma in [0.3, 0.5]:
        lut = np.array([((i/255.0)**gamma)*255 for i in range(256)], np.uint8)
        variants.append(cv2.bitwise_not(cv2.LUT(zone_up, lut)))
    # tophat
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
    variants.append(cv2.morphologyEx(zone_up, cv2.MORPH_TOPHAT, kern))

    # Stack vertically with labels
    labeled_strips = []
    labels = ["C4","C8","C16","C32","C64","G0.3","G0.5","TopH"]
    for lbl, var in zip(labels, variants):
        v = var.copy()
        cv2.putText(v, lbl, (3, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128.0,), 1)
        labeled_strips.append(v)

    max_w_strip = max(s.shape[1] for s in labeled_strips)
    padded = []
    for s in labeled_strips:
        if s.shape[1] < max_w_strip:
            pad = np.zeros((s.shape[0], max_w_strip - s.shape[1]), dtype=np.uint8)
            s = np.hstack([s, pad])
        padded.append(s)
    multi_enh = np.vstack(padded)
    cv2.imwrite(f"/tmp/pos{pos}_multi_enhance.png", multi_enh)

print("\n\nDebug images saved to /tmp/pos*_6x_*.png and /tmp/pos*_multi_enhance.png")


# =============================================================================
# Summary: best structural match per position
# =============================================================================
print("\n" + "=" * 70)
print("SUMMARY: Best structural match per hidden position")
print("=" * 70)

for pos in range(6, 12):
    zone = get_zone(pos)
    zone_up = upscale(zone)
    enh = best_enhance(zone_up)
    bn = binarize(enh)
    f = features(bn)

    distances = {}
    for digit, avg_f in avg_features.items():
        keys_weights = [
            ("TL", 1), ("TR", 2), ("BL", 1), ("BR", 2),
            ("gap_r", 15), ("gap_l", 10),
            ("right_coverage", 80),
            ("h_cross", 20), ("v_cross", 15),
            ("t_right", 1.5), ("m_right", 1.5), ("b_right", 1.5),
        ]
        dist = sum(w * abs(f[k] - avg_f[k]) for k, w in keys_weights)
        distances[digit] = dist

    ranked = sorted(distances.items(), key=lambda x: x[1])
    top3 = ", ".join(f"'{d}'({v:.0f})" for d, v in ranked[:3])
    print(f"  POS {pos}: {top3}")

    # Key observations
    obs = []
    if f["gap_r"] > 5:
        obs.append(f"right_gap={f['gap_r']}")
    if f["gap_l"] > 5:
        obs.append(f"left_gap={f['gap_l']}")
    if f["right_coverage"] < 0.5:
        obs.append("sparse_right")
    if f["right_coverage"] > 0.8:
        obs.append("full_right")
    if f["TR"] < 30:
        obs.append("empty_TR")
    if f["BR"] < 30:
        obs.append("empty_BR")
    if obs:
        print(f"         Observations: {', '.join(obs)}")
