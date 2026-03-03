#!/usr/bin/env python3
"""
Ultra-detailed pixel structure analysis for POSITION 11.

User says: "I don't see a '2' before 0665, there's a space where the
right curve should be."

This script:
1. Extracts pos 11 at 6x resolution with multiple enhancements
2. Splits into left/right halves — analyses intensity per column
3. Compares right-half profiles against ALL known digits (0,3,4,5,6,8)
4. Checks for the presence/absence of a right-side curve explicitly
5. Does column-by-column brightness analysis of embossed strokes
6. Generates visual debug images with annotations
7. Re-OCR with even narrower crops focusing on the character shape
8. Also checks neighboring position 10 for comparison
"""
from __future__ import annotations

import re
from collections import Counter

import cv2
import numpy as np
import pytesseract

IMG = "/Users/jenineferderas/Desktop/card_image.jpg"
WL = "-c tessedit_char_whitelist=0123456789"
OCR_EXCEPTIONS = (
    pytesseract.TesseractError,
    RuntimeError,
    TypeError,
    ValueError,
)
OCR_TIMEOUT_SEC = 0.8
FAST_OCR_SWEEP = True
MAX_OCR_CALLS_PER_POS = 1200 if FAST_OCR_SWEEP else 4000

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
KNOWN_DIGITS = {
    0: "4", 1: "3", 2: "8", 3: "8", 4: "5", 5: "4",
    12: "0", 13: "6", 14: "6", 15: "5",
}
HALF = 34
SCALE = 6  # Higher res for detailed analysis


def upscale(im, factor=SCALE):
    return cv2.resize(im, (im.shape[1]*factor, im.shape[0]*factor),
                      interpolation=cv2.INTER_CUBIC)


def get_zone(pos, dx=0):
    cx = CENTERS[pos] + dx
    x0 = max(0, cx - HALF)
    x1 = min(rw, cx + HALF)
    return roi_gray[:, x0:x1]


def enhance_clahe(zone, clip=16):
    c = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(3, 3))
    return cv2.bitwise_not(c.apply(zone))


def enhance_gamma(zone, gamma=0.3):
    lut = np.array([((i/255.0)**gamma)*255 for i in range(256)], np.uint8)
    return cv2.bitwise_not(cv2.LUT(zone, lut))


def enhance_tophat(zone):
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    return cv2.morphologyEx(zone, cv2.MORPH_TOPHAT, kern)


def enhance_unsharp(zone):
    blur = cv2.GaussianBlur(zone, (0, 0), 3)
    return cv2.bitwise_not(cv2.addWeighted(zone, 2.5, blur, -1.5, 0))


# =============================================================================
# STEP 1: Visual column-by-column profile of position 11
# =============================================================================
print("=" * 65)
print("POSITION 11 — DETAILED PIXEL STRUCTURE ANALYSIS")
print("=" * 65)

# Try multiple small offsets around the center
for dx in [-5, -3, 0, 3, 5]:
    zone_raw = get_zone(11, dx)
    zone_up = upscale(zone_raw)

    # Best enhancements for embossed text
    enh_clahe = enhance_clahe(zone_up, 16)
    enh_gamma = enhance_gamma(zone_up, 0.3)
    enh_tophat = enhance_tophat(zone_up)
    enh_unsharp = enhance_unsharp(zone_up)

    # Binarize with Otsu to isolate strokes
    _, binary = cv2.threshold(enh_clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    zh, zw = binary.shape
    mid_col = zw // 2

    # Column intensity profiles
    col_means = binary.astype(np.float64).mean(axis=0)

    # Split into left (0..mid) and right (mid..end)
    left_mean = col_means[:mid_col].mean()
    right_mean = col_means[mid_col:].mean()

    # Row profiles for top/middle/bottom thirds
    third = zh // 3
    top_row_mean = binary[:third].astype(np.float64).mean(axis=1)
    mid_row_mean = binary[third:2*third].astype(np.float64).mean(axis=1)
    bot_row_mean = binary[2*third:].astype(np.float64).mean(axis=1)

    # Right-half: find columns with significant white pixels (stroke presence)
    right_cols = binary[:, mid_col:]
    right_col_means = right_cols.astype(np.float64).mean(axis=0)

    # Check top-right quadrant vs bottom-right quadrant
    tr_density = binary[:zh//2, mid_col:].astype(np.float64).mean()
    br_density = binary[zh//2:, mid_col:].astype(np.float64).mean()

    # Check for "gap" in right side — consecutive low-density columns
    threshold_for_gap = 30  # columns with mean < this have no stroke
    gap_cols = right_col_means < threshold_for_gap
    gap_runs = []
    run_start = None
    for i, is_gap in enumerate(gap_cols):
        if is_gap and run_start is None:
            run_start = i
        elif not is_gap and run_start is not None:
            gap_runs.append((run_start, i - 1, i - run_start))
            run_start = None
    if run_start is not None:
        gap_runs.append((run_start, len(gap_cols) - 1, len(gap_cols) - run_start))

    if dx == 0:
        print(f"\n  dx={dx} — PRIMARY analysis")
    else:
        print(f"\n  dx={dx}")
    print(f"    Zone: {zw}x{zh} pixels (after {SCALE}x upscale)")
    print(f"    Left-half avg brightness:  {left_mean:.1f}")
    print(f"    Right-half avg brightness: {right_mean:.1f}")
    print(f"    Top-right density:    {tr_density:.1f}")
    print(f"    Bottom-right density: {br_density:.1f}")
    print("    Right-side gap runs (col_offset, end, length):")
    for start, end, length in gap_runs:
        if length > 3:
            print(f"      col {start}-{end} (length={length}) {'*** SIGNIFICANT GAP ***' if length > 10 else ''}")

    if dx == 0:
        # Save detailed debug images
        cv2.imwrite("/tmp/pos11_binary_otsu.png", binary)
        cv2.imwrite("/tmp/pos11_clahe16_6x.png", enh_clahe)
        cv2.imwrite("/tmp/pos11_gamma03_6x.png", enh_gamma)
        cv2.imwrite("/tmp/pos11_tophat_6x.png", enh_tophat)
        cv2.imwrite("/tmp/pos11_unsharp_6x.png", enh_unsharp)

        # Draw column profile as image
        profile_img = np.zeros((256, zw), dtype=np.uint8)
        for x, val in enumerate(col_means):
            y_bar = int(min(val, 255))
            profile_img[256-y_bar:, x] = 255
        # Draw midline
        cv2.line(profile_img, (mid_col, 0), (mid_col, 255), (128.0,), 1)
        cv2.imwrite("/tmp/pos11_column_profile.png", profile_img)

        # Draw row profile
        row_means = binary.astype(np.float64).mean(axis=1)
        row_prof_img = np.zeros((zh, 256), dtype=np.uint8)
        for y, val in enumerate(row_means):
            x_bar = int(min(val, 255))
            row_prof_img[y, :x_bar] = 255
        cv2.imwrite("/tmp/pos11_row_profile.png", row_prof_img)


# =============================================================================
# STEP 2: Same analysis for ALL known digits for comparison
# =============================================================================
print("\n" + "=" * 65)
print("COMPARISON: Right-side structure of ALL known digits")
print("=" * 65)

for kpos, kdigit in sorted(KNOWN_DIGITS.items()):
    zone_raw = get_zone(kpos)
    zone_up = upscale(zone_raw)
    enh = enhance_clahe(zone_up, 16)
    _, binary = cv2.threshold(enh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    zh, zw = binary.shape
    mid_col = zw // 2

    left_mean = binary[:, :mid_col].astype(np.float64).mean()
    right_mean = binary[:, mid_col:].astype(np.float64).mean()
    tr_density = binary[:zh//2, mid_col:].astype(np.float64).mean()
    br_density = binary[zh//2:, mid_col:].astype(np.float64).mean()

    right_cols = binary[:, mid_col:]
    right_col_means = right_cols.astype(np.float64).mean(axis=0)
    gap_cols = right_col_means < 30
    max_gap = 0
    run = 0
    for g in gap_cols:
        if g:
            run += 1
            max_gap = max(max_gap, run)
        else:
            run = 0

    print(f"  pos{kpos:2d}='{kdigit}': L={left_mean:.0f} R={right_mean:.0f} "
          f"TR={tr_density:.0f} BR={br_density:.0f} maxRightGap={max_gap}")


# =============================================================================
# STEP 3: Side-by-side visual comparison
# =============================================================================
print("\n" + "=" * 65)
print("STEP 3: Creating enhanced comparison strips")
print("=" * 65)

# Create comparison of pos11 vs each known digit at multiple enhancements
for enh_name, enh_fn in [("clahe16", lambda z: enhance_clahe(z, 16)),
                          ("gamma03", lambda z: enhance_gamma(z, 0.3)),
                          ("tophat", enhance_tophat),
                          ("unsharp", enhance_unsharp)]:
    panels = []
    # Known digits first
    for kpos in sorted(KNOWN_DIGITS.keys()):
        zone = get_zone(kpos)
        zone_up = upscale(zone)
        enh = enh_fn(zone_up)
        labeled = enh.copy()
        cv2.putText(
            labeled,
            f"{KNOWN_DIGITS[kpos]}",
            (5, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (128.0,),
            2,
        )
        panels.append(labeled)
    # Target pos 11
    zone11 = get_zone(11)
    zone11_up = upscale(zone11)
    enh11 = enh_fn(zone11_up)
    labeled11 = enh11.copy()
    cv2.putText(
        labeled11,
        "?11",
        (5, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (128.0,),
        2,
    )
    panels.append(labeled11)

    # Also target pos 10 for context
    zone10 = get_zone(10)
    zone10_up = upscale(zone10)
    enh10 = enh_fn(zone10_up)
    labeled10 = enh10.copy()
    cv2.putText(
        labeled10,
        "?10",
        (5, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (128.0,),
        2,
    )
    panels.append(labeled10)

    max_h = max(p.shape[0] for p in panels)
    max_w = max(p.shape[1] for p in panels)
    padded = []
    for p in panels:
        ph, pw = p.shape[:2]
        canvas = np.zeros((max_h, max_w), dtype=np.uint8)
        canvas[:ph, :pw] = p
        padded.append(canvas)
    strip = np.hstack(padded)
    cv2.imwrite(f"/tmp/pos11_vs_known_{enh_name}.png", strip)
    print(f"  Saved: /tmp/pos11_vs_known_{enh_name}.png")


# =============================================================================
# STEP 4: Re-OCR with TIGHTER crop around the digit (remove background noise)
# =============================================================================
print("\n" + "=" * 65)
print("STEP 4: OCR with tighter vertical crops")
print("=" * 65)

# The embossed digits occupy roughly the middle 60% of the ROI height
# Try tighter vertical windows to reduce noise

CONFIGS = [
    f"--oem 3 --psm 10 {WL}",
    f"--oem 3 --psm 13 {WL}",
]
if not FAST_OCR_SWEEP:
    CONFIGS.append(f"--oem 3 --psm 8 {WL}")
THRESHOLDS = [120, 150] if FAST_OCR_SWEEP else [110, 130, 150]


def ocr_digit(im, budget):
    reads = []
    calls_used = 0
    if budget <= 0:
        return reads, calls_used
    for cfg in CONFIGS:
        if calls_used >= budget:
            break
        try:
            txt = pytesseract.image_to_string(
                im,
                config=cfg,
                timeout=OCR_TIMEOUT_SEC,
            ).strip()
            d = re.sub(r"\D", "", txt)
            if d:
                reads.append(d[0])
        except OCR_EXCEPTIONS:
            pass
        calls_used += 1
        for thr in THRESHOLDS:
            if calls_used >= budget:
                break
            _, bw = cv2.threshold(im, thr, 255, cv2.THRESH_BINARY)
            try:
                txt = pytesseract.image_to_string(
                    bw,
                    config=cfg,
                    timeout=OCR_TIMEOUT_SEC,
                ).strip()
                d = re.sub(r"\D", "", txt)
                if d:
                    reads.append(d[0])
            except OCR_EXCEPTIONS:
                pass
            calls_used += 1
    return reads, calls_used


# Different vertical crop strategies
v_crops = [
    ("full", 0.0, 1.0),
    ("mid70", 0.15, 0.85),
    ("mid50", 0.25, 0.75),
]
if not FAST_OCR_SWEEP:
    v_crops.extend([("top60", 0.0, 0.6), ("bot60", 0.4, 1.0)])

for target_pos in [11, 10]:
    print(f"\n--- POS {target_pos} tight-crop OCR ---")
    all_votes = Counter()
    budget_remaining = MAX_OCR_CALLS_PER_POS

    for crop_name, vy0_frac, vy1_frac in v_crops:
        if budget_remaining <= 0:
            break
        crop_votes = Counter()

        dx_offsets = [-3, 0, 3] if FAST_OCR_SWEEP else [-5, -2, 0, 2, 5]
        for dx in dx_offsets:
            if budget_remaining <= 0:
                break
            cx = CENTERS[target_pos] + dx
            x0 = max(0, cx - HALF)
            x1 = min(rw, cx + HALF)
            zone = roi_gray[:, x0:x1]

            # Apply vertical crop
            zh = zone.shape[0]
            vy0 = int(zh * vy0_frac)
            vy1 = int(zh * vy1_frac)
            cropped = zone[vy0:vy1, :]
            cropped_up = upscale(cropped)

            enh_fns = [
                lambda z: enhance_clahe(z, 8),
                lambda z: enhance_clahe(z, 16),
                lambda z: enhance_clahe(z, 32),
                lambda z: enhance_gamma(z, 0.3),
                enhance_tophat,
                enhance_unsharp,
            ]
            if FAST_OCR_SWEEP:
                enh_fns = enh_fns[:4]
            for enh_fn in enh_fns:
                if budget_remaining <= 0:
                    break
                enh = enh_fn(cropped_up)
                reads, calls_used = ocr_digit(enh, budget_remaining)
                budget_remaining -= calls_used
                for d in reads:
                    crop_votes[d] += 1
                    all_votes[d] += 1

        total_crop = sum(crop_votes.values())
        if total_crop > 0:
            top3 = crop_votes.most_common(3)
            summary = ", ".join(f"'{d}'={n}" for d, n in top3)
            print(f"  {crop_name:>6}: {summary}  (n={total_crop})")

    total = sum(all_votes.values())
    calls_done = MAX_OCR_CALLS_PER_POS - budget_remaining
    print(f"  OCR calls used: {calls_done}/{MAX_OCR_CALLS_PER_POS}")
    if budget_remaining <= 0:
        print("  NOTE: OCR call budget reached; using current vote totals.")
    print(f"\n  COMBINED: {total} reads")
    for d, n in all_votes.most_common(6):
        pct = n / total * 100 if total else 0
        bar = "#" * max(1, int(pct / 2))
        print(f"    '{d}': {n:4d} ({pct:5.1f}%)  {bar}")

    if all_votes:
        top2 = all_votes.most_common(2)
        print(f"\n  >>> POS {target_pos} verdict: '{top2[0][0]}' ({top2[0][1]/total:.0%})")
        if len(top2) > 1:
            print(f"      Runner-up: '{top2[1][0]}' ({top2[1][1]/total:.0%})")


# =============================================================================
# STEP 5: Curvature analysis — detect if there's a right curve vs straight line
# =============================================================================
print("\n" + "=" * 65)
print("STEP 5: Curvature detection — right-side stroke analysis")
print("=" * 65)

for target_pos in [11]:
    zone = get_zone(target_pos)
    zone_up = upscale(zone)
    enh = enhance_clahe(zone_up, 16)

    # Binarize
    _, binary = cv2.threshold(enh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    zh, zw = binary.shape
    mid_col = zw // 2

    # For each row in the right half, find the rightmost white pixel
    # This traces the outer contour of the right side
    rightmost_x = []
    leftmost_right_x = []  # leftmost white pixel in right half
    for y in range(zh):
        row = binary[y, mid_col:]
        white_cols = np.nonzero(row > 128)[0]
        if len(white_cols) > 0:
            rightmost_x.append(white_cols[-1])
            leftmost_right_x.append(white_cols[0])
        else:
            rightmost_x.append(-1)
            leftmost_right_x.append(-1)

    rightmost_x = np.array(rightmost_x)
    leftmost_right_x = np.array(leftmost_right_x)

    # Smooth the contour
    valid_mask = rightmost_x >= 0
    if valid_mask.sum() > 10:
        # Contour analysis
        valid_y = np.nonzero(valid_mask)[0]
        valid_rx = rightmost_x[valid_mask].astype(np.float64)

        # Check if contour curves (2nd derivative) vs straight
        if len(valid_rx) > 5:
            smoothed = cv2.GaussianBlur(valid_rx.reshape(-1, 1), (21, 1), 0).flatten()
            dx = np.gradient(smoothed)
            ddx = np.gradient(dx)

            # Average curvature
            avg_curv = np.mean(np.abs(ddx))
            max_curv = np.max(np.abs(ddx))

            # Curvature in top third vs middle vs bottom third
            third = len(ddx) // 3
            top_curv = np.mean(np.abs(ddx[:third]))
            mid_curv = np.mean(np.abs(ddx[third:2*third]))
            bot_curv = np.mean(np.abs(ddx[2*third:]))

            print(f"\n  POS {target_pos} — Right-side outer contour:")
            print(f"    Avg curvature:    {avg_curv:.4f}")
            print(f"    Max curvature:    {max_curv:.4f}")
            print(f"    Top 1/3 curv:     {top_curv:.4f}")
            print(f"    Middle 1/3 curv:  {mid_curv:.4f}")
            print(f"    Bottom 1/3 curv:  {bot_curv:.4f}")

            # Width of stroke in right half
            right_widths = rightmost_x[valid_mask] - leftmost_right_x[valid_mask]
            print(f"    Avg right-stroke width: {np.mean(right_widths):.1f} px")
            print(f"    Max right-stroke width: {np.max(right_widths)} px")

            # For digit identification:
            # '2' has a curve at top-right, then sweeps left → rightmost_x decreases from top to bottom
            # '7' has a top-right origin, slopes left → rightmost_x also decreases
            # '3' curves right in top AND bottom → rightmost_x peaks twice
            # '8' curves right in both halves → rightmost_x peaks twice
            # '0' nearly circular → rightmost_x peaks in middle
            # '4' has a vertical right stroke → rightmost_x fairly constant

            # Check monotonicity
            slope_top_half = np.polyfit(np.arange(len(smoothed[:len(smoothed)//2])),
                                         smoothed[:len(smoothed)//2], 1)[0]
            slope_bot_half = np.polyfit(np.arange(len(smoothed[len(smoothed)//2:])),
                                         smoothed[len(smoothed)//2:], 1)[0]

            print(f"    Right contour slope (top half):    {slope_top_half:+.3f}")
            print(f"    Right contour slope (bottom half): {slope_bot_half:+.3f}")

            # Interpretation
            print("\n    INTERPRETATION:")
            if slope_top_half < -0.05 and slope_bot_half < -0.05:
                print("      Rightmost edge moves LEFT through both halves")
                print("      -> Consistent with '2' or '7' (both slope left going down)")
            elif slope_top_half > 0.02 and slope_bot_half < -0.02:
                print("      Bulge at top-right -> Could be '3' top loop")
            elif abs(slope_top_half) < 0.02 and abs(slope_bot_half) < 0.02:
                print("      Fairly vertical right edge -> Consistent with '4', '1'")
            else:
                print(f"      Top slope={slope_top_half:+.3f}, Bot slope={slope_bot_half:+.3f}")

    # Also draw the contour on the image
    contour_img = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    for y in range(zh):
        if rightmost_x[y] >= 0:
            x = mid_col + rightmost_x[y]
            if 0 <= x < zw:
                contour_img[y, x] = (0, 0, 255)  # Red dot
        if leftmost_right_x[y] >= 0:
            x = mid_col + leftmost_right_x[y]
            if 0 <= x < zw:
                contour_img[y, x] = (0, 255, 0)  # Green dot
    cv2.line(contour_img, (mid_col, 0), (mid_col, zh-1), (255, 0, 0), 1)
    cv2.imwrite("/tmp/pos11_contour_analysis.png", contour_img)
    print("\n  Saved: /tmp/pos11_contour_analysis.png")
    print("  (Red = rightmost edge, Green = leftmost-in-right-half, Blue = midline)")


# =============================================================================
# STEP 6: Do the SAME contour analysis for known digits for reference
# =============================================================================
print("\n" + "=" * 65)
print("REFERENCE: Right-side contour slopes for known digits")
print("=" * 65)

for kpos, kdigit in sorted(KNOWN_DIGITS.items()):
    zone = get_zone(kpos)
    zone_up = upscale(zone)
    enh = enhance_clahe(zone_up, 16)
    _, binary = cv2.threshold(enh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    zh, zw = binary.shape
    mid_col = zw // 2

    rightmost_x = []
    for y in range(zh):
        row = binary[y, mid_col:]
        white_cols = np.nonzero(row > 128)[0]
        if len(white_cols) > 0:
            rightmost_x.append(white_cols[-1])
        else:
            rightmost_x.append(-1)

    rx = np.array(rightmost_x)
    valid = rx >= 0
    if valid.sum() > 10:
        vrx = rx[valid].astype(np.float64)
        smoothed = cv2.GaussianBlur(vrx.reshape(-1, 1), (21, 1), 0).flatten()
        half = len(smoothed) // 2
        slope_top = np.polyfit(np.arange(half), smoothed[:half], 1)[0]
        slope_bot = np.polyfit(np.arange(len(smoothed) - half), smoothed[half:], 1)[0]
        avg_curv = np.mean(np.abs(np.gradient(np.gradient(smoothed))))
        print(f"  pos{kpos:2d}='{kdigit}': top_slope={slope_top:+.3f}, "
              f"bot_slope={slope_bot:+.3f}, curv={avg_curv:.4f}")


print("\n" + "=" * 65)
print("ANALYSIS COMPLETE — Check /tmp/pos11_*.png for visual debug")
print("=" * 65)
