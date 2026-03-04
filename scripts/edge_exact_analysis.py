    #!/usr/bin/env python3
"""
Exact edge-based observation of hidden positions 6-11.

The marker occlusion darkens the whole region, making raw brightness
unreliable. Instead, we use:
  1. R-channel extraction (red light penetrates dark inks better)
  2. Edge detection (Canny, Sobel) — shape survives through marker
  3. Row-by-row and column-by-column edge density profiles
  4. Contour analysis: count, bounding boxes, connected components
  5. Very detailed per-column white-pixel analysis for RIGHT side gaps

For each position: print ASCII art of the edge map for direct inspection.
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
roi_r = roi_bgr[:, :, 2]  # Red channel
rh, rw = roi_gray.shape

CENTERS = {
    0: 197,  1: 272,  2: 347,  3: 422,
    4: 497,  5: 572,  6: 647,  7: 722,
    8: 833,  9: 908,  10: 983, 11: 1058,
    12: 1133, 13: 1208, 14: 1283, 15: 1358,
}
KNOWN = {0:"4", 1:"3", 2:"8", 3:"8", 4:"5", 5:"4", 12:"0", 13:"6", 14:"6", 15:"5"}
HALF = 34
SCALE = 4


def extract_zone_gray(pos, dx=0):
    cx = CENTERS[pos] + dx
    x0, x1 = max(0, cx - HALF), min(rw, cx + HALF)
    z = roi_gray[:, x0:x1]
    return cv2.resize(z, (z.shape[1]*SCALE, z.shape[0]*SCALE), interpolation=cv2.INTER_CUBIC)


def extract_zone_r(pos, dx=0):
    cx = CENTERS[pos] + dx
    x0, x1 = max(0, cx - HALF), min(rw, cx + HALF)
    z = roi_r[:, x0:x1]
    return cv2.resize(z, (z.shape[1]*SCALE, z.shape[0]*SCALE), interpolation=cv2.INTER_CUBIC)


def enhance_for_edges(zone):
    """Multiple enhancements to extract edges through marker occlusion."""
    results = []
    
    # CLAHE inverted (strong)
    for clip in [16, 32, 64]:
        enh = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(3,3)).apply(zone)
        results.append(cv2.bitwise_not(enh))
    
    # Gamma 0.3 (very bright) inverted
    lut = np.array([((i/255.0)**0.3)*255 for i in range(256)], np.uint8)
    results.append(cv2.bitwise_not(cv2.LUT(zone, lut)))
    
    # Top-hat
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    results.append(cv2.morphologyEx(zone, cv2.MORPH_TOPHAT, kern))
    
    # Unsharp + CLAHE
    usm = cv2.addWeighted(zone, 2.5, cv2.GaussianBlur(zone, (0,0), 3), -1.5, 0)
    results.append(cv2.bitwise_not(cv2.createCLAHE(clipLimit=32.0, tileGridSize=(3,3)).apply(usm)))
    
    return results


def get_edge_map(zone, low=30, high=90):
    """Aggregate edge map from multiple enhancements."""
    enhancements = enhance_for_edges(zone)
    composite = np.zeros(zone.shape[:2], dtype=np.float64)
    for enh in enhancements:
        edges = cv2.Canny(enh, low, high)
        composite += edges.astype(np.float64)
    # Normalize
    if composite.max() > 0:
        composite = (composite / composite.max() * 255).astype(np.uint8)
    _, bw = cv2.threshold(composite, 40, 255, cv2.THRESH_BINARY)
    return bw


def edge_features(edge_map):
    """Extract structural features from edge map."""
    zh, zw = edge_map.shape
    margin_x = int(zw * 0.08)
    margin_y = int(zh * 0.08)
    crop = edge_map[margin_y:zh-margin_y, margin_x:zw-margin_x]
    ch, cw = crop.shape
    cmh, cmw = ch // 2, cw // 2
    th = ch // 3

    # Quadrant edge density
    tl = crop[:cmh, :cmw].astype(float).mean()
    tr = crop[:cmh, cmw:].astype(float).mean()
    bl = crop[cmh:, :cmw].astype(float).mean()
    br = crop[cmh:, cmw:].astype(float).mean()

    # 6-zone
    t_l = crop[:th, :cmw].astype(float).mean()
    t_r = crop[:th, cmw:].astype(float).mean()
    m_l = crop[th:2*th, :cmw].astype(float).mean()
    m_r = crop[th:2*th, cmw:].astype(float).mean()
    b_l = crop[2*th:, :cmw].astype(float).mean()
    b_r = crop[2*th:, cmw:].astype(float).mean()

    # Column-wise edge density in right half
    right_cols = crop[:, cmw:]
    col_density = right_cols.astype(float).mean(axis=0) / 255.0

    # Right gap: longest run of low-density columns
    gap_r = 0
    run = 0
    for v in col_density:
        if v < 0.05:
            run += 1
            gap_r = max(gap_r, run)
        else:
            run = 0

    # Left gap
    left_cols = crop[:, :cmw]
    col_density_l = left_cols.astype(float).mean(axis=0) / 255.0
    gap_l = 0
    run = 0
    for v in col_density_l:
        if v < 0.05:
            run += 1
            gap_l = max(gap_l, run)
        else:
            run = 0

    # Row coverage: fraction of rows with ANY edge in right half
    row_has_edge_r = (right_cols.astype(float).max(axis=1) > 0).mean()
    row_has_edge_l = (left_cols.astype(float).max(axis=1) > 0).mean()

    # Top-third right density vs bottom-third right density
    t_r_dens = crop[:th, cmw:].astype(float).mean()
    b_r_dens = crop[2*th:, cmw:].astype(float).mean()

    # Mid crossings
    mid_row = crop[cmh-2:cmh+2, :].astype(float).mean(axis=0)
    h_cross = int(np.sum(np.abs(np.diff((mid_row > 30).astype(int)))))

    mid_col = crop[:, cmw-2:cmw+2].astype(float).mean(axis=1)
    v_cross = int(np.sum(np.abs(np.diff((mid_col > 30).astype(int)))))

    # Overall edge density
    total = crop.astype(float).mean()

    # Connected components
    num_cc, _ = cv2.connectedComponents(crop)
    num_cc -= 1  # subtract background

    return {
        "TL": tl, "TR": tr, "BL": bl, "BR": br,
        "t_l": t_l, "t_r": t_r, "m_l": m_l, "m_r": m_r,
        "b_l": b_l, "b_r": b_r,
        "gap_r": gap_r, "gap_l": gap_l,
        "row_R": row_has_edge_r, "row_L": row_has_edge_l,
        "t_r_dens": t_r_dens, "b_r_dens": b_r_dens,
        "h_cross": h_cross, "v_cross": v_cross,
        "density": total, "cc": num_cc,
    }


def ascii_art(edge_map, max_w=60, max_h=30):
    """Create ASCII representation of edge map for terminal inspection."""
    zh, zw = edge_map.shape
    # Downsample
    step_y = max(1, zh // max_h)
    step_x = max(1, zw // max_w)
    lines = []
    for y in range(0, zh, step_y):
        row = ""
        for x in range(0, zw, step_x):
            block = edge_map[y:y+step_y, x:x+step_x]
            val = block.astype(float).mean()
            if val > 150:
                row += "█"
            elif val > 80:
                row += "▓"
            elif val > 30:
                row += "░"
            else:
                row += " "
        lines.append(row)
    return "\n".join(lines)


# =============================================================================
print(f"Image: {w}x{h}, ROI: {rw}x{rh}, Scale: {SCALE}x\n")

# Build known digit edge features
print("=" * 70)
print("KNOWN DIGITS — EDGE FEATURES")
print("=" * 70)

digit_edge_features: dict[str, list[dict]] = defaultdict(list)

hdr = (f"{'P':>2} {'D':>1} {'TL':>5} {'TR':>5} {'BL':>5} {'BR':>5} "
       f"{'gR':>3} {'gL':>3} {'rR':>4} {'Hx':>2} {'Vx':>2} {'CC':>3} {'dens':>5}")
print(hdr)
print("-" * 70)

for pos in sorted(KNOWN.keys()):
    digit = KNOWN[pos]
    # Use GRAY for known (no marker), also try R-channel
    zone_gray = extract_zone_gray(pos)
    zone_r = extract_zone_r(pos)
    
    edge_g = get_edge_map(zone_gray)
    edge_r = get_edge_map(zone_r)
    # Combine
    edge_combined = cv2.bitwise_or(edge_g, edge_r)
    
    f = edge_features(edge_combined)
    digit_edge_features[digit].append(f)
    
    print(f"{pos:>2} {digit:>1} {f['TL']:5.1f} {f['TR']:5.1f} {f['BL']:5.1f} {f['BR']:5.1f} "
          f"{f['gap_r']:>3} {f['gap_l']:>3} {f['row_R']:4.2f} "
          f"{f['h_cross']:>2} {f['v_cross']:>2} {f['cc']:>3} {f['density']:5.1f}")

# Average per digit
avg_ef: dict[str, dict] = {}
for digit, flist in digit_edge_features.items():
    avg = {}
    for key in flist[0]:
        avg[key] = np.mean([f[key] for f in flist])
    avg_ef[digit] = avg

print("\nDigit averages:")
for digit in sorted(avg_ef.keys()):
    f = avg_ef[digit]
    print(f"   {digit:>1} {f['TL']:5.1f} {f['TR']:5.1f} {f['BL']:5.1f} {f['BR']:5.1f} "
          f"{f['gap_r']:>3.0f} {f['gap_l']:>3.0f} {f['row_R']:4.2f} "
          f"{f['h_cross']:>2.0f} {f['v_cross']:>2.0f} {f['cc']:>3.0f} {f['density']:5.1f}")

# Show ASCII art of known digits for reference
print("\n" + "=" * 70)
print("KNOWN DIGITS — EDGE ASCII ART")
print("=" * 70)

for pos in sorted(KNOWN.keys()):
    digit = KNOWN[pos]
    zone = extract_zone_gray(pos)
    edges = get_edge_map(zone)
    print(f"\n--- pos{pos} = '{digit}' ---")
    print(ascii_art(edges, max_w=36, max_h=18))


# =============================================================================
print("\n" + "=" * 70)
print("HIDDEN POSITIONS — EDGE ANALYSIS + ASCII ART")
print("=" * 70)

for pos in range(6, 12):
    zone_gray = extract_zone_gray(pos)
    zone_r = extract_zone_r(pos)
    
    # For hidden positions, R-channel is KEY (penetrates marker)
    edge_g = get_edge_map(zone_gray)
    edge_r = get_edge_map(zone_r)
    edge_combined = cv2.bitwise_or(edge_g, edge_r)
    
    # Also try with higher CLAHE and lower Canny thresholds (more sensitive)
    edge_sensitive = get_edge_map(zone_r, low=15, high=50)
    edge_final = cv2.bitwise_or(edge_combined, edge_sensitive)
    
    f = edge_features(edge_final)
    
    print(f"\n{'━' * 70}")
    print(f"POSITION {pos} (cx={CENTERS[pos]})")
    print(f"{'━' * 70}")
    print(f"  TL={f['TL']:.1f}  TR={f['TR']:.1f}  BL={f['BL']:.1f}  BR={f['BR']:.1f}")
    print(f"  gap_R={f['gap_r']}  gap_L={f['gap_l']}  row_R={f['row_R']:.2f}  row_L={f['row_L']:.2f}")
    print(f"  h_cross={f['h_cross']}  v_cross={f['v_cross']}  CC={f['cc']}  density={f['density']:.1f}")
    print(f"  6-zone: tL={f['t_l']:.0f} tR={f['t_r']:.0f} mL={f['m_l']:.0f} mR={f['m_r']:.0f} bL={f['b_l']:.0f} bR={f['b_r']:.0f}")
    print(f"  top-right vs bot-right: {f['t_r_dens']:.1f} vs {f['b_r_dens']:.1f}")
    
    # ASCII art of the edge map
    print("\n  Edge map (gray-based):")
    print("  " + ascii_art(edge_g, max_w=46, max_h=20).replace("\n", "\n  "))
    
    print("\n  Edge map (R-channel, more sensitive):")
    print("  " + ascii_art(edge_sensitive, max_w=46, max_h=20).replace("\n", "\n  "))
    
    print("\n  Combined edge map:")
    print("  " + ascii_art(edge_final, max_w=46, max_h=20).replace("\n", "\n  "))
    
    # Distance to known digits
    print("\n  Similarity to known digits (edge-based, lower=better):")
    distances = {}
    for digit, avg_f in sorted(avg_ef.items()):
        weights = [
            ("TL", 1), ("TR", 2), ("BL", 1), ("BR", 2),
            ("gap_r", 8), ("gap_l", 5),
            ("row_R", 50), ("row_L", 30),
            ("t_r_dens", 0.8), ("b_r_dens", 0.8),
            ("h_cross", 12), ("v_cross", 8),
            ("cc", 5), ("density", 0.5),
        ]
        dist = sum(w * abs(f[k] - avg_f[k]) for k, w in weights)
        distances[digit] = dist

    ranked = sorted(distances.items(), key=lambda x: x[1])
    for i, (d, dist) in enumerate(ranked):
        marker = " ◄◄◄" if i == 0 else ""
        print(f"    '{d}': {dist:8.1f}{marker}")

    # Save images
    cv2.imwrite(f"/tmp/pos{pos}_edge_gray.png", edge_g)
    cv2.imwrite(f"/tmp/pos{pos}_edge_r.png", edge_r)
    cv2.imwrite(f"/tmp/pos{pos}_edge_combined.png", edge_final)
    
    # Save enhancements
    for i, enh in enumerate(enhance_for_edges(zone_r)):
        cv2.imwrite(f"/tmp/pos{pos}_r_enh{i}.png", enh)


# =============================================================================
print("\n\n" + "=" * 70)
print("FINAL SUMMARY — EDGE-BASED BEST MATCH")
print("=" * 70)

for pos in range(6, 12):
    zone_gray = extract_zone_gray(pos)
    zone_r = extract_zone_r(pos)
    edge_g = get_edge_map(zone_gray)
    edge_r = get_edge_map(zone_r)
    edge_combined = cv2.bitwise_or(edge_g, edge_r)
    edge_sensitive = get_edge_map(zone_r, low=15, high=50)
    edge_final = cv2.bitwise_or(edge_combined, edge_sensitive)
    f = edge_features(edge_final)

    distances = {}
    for digit, avg_f in avg_ef.items():
        weights = [
            ("TL", 1), ("TR", 2), ("BL", 1), ("BR", 2),
            ("gap_r", 8), ("gap_l", 5),
            ("row_R", 50), ("row_L", 30),
            ("t_r_dens", 0.8), ("b_r_dens", 0.8),
            ("h_cross", 12), ("v_cross", 8),
            ("cc", 5), ("density", 0.5),
        ]
        dist = sum(w * abs(f[k] - avg_f[k]) for k, w in weights)
        distances[digit] = dist

    ranked = sorted(distances.items(), key=lambda x: x[1])
    top3 = ", ".join(f"'{d}'({v:.0f})" for d, v in ranked[:3])
    
    # Observations
    obs = []
    if f["gap_r"] > 3: obs.append(f"RIGHT_GAP={f['gap_r']}")
    if f["gap_l"] > 3: obs.append(f"LEFT_GAP={f['gap_l']}")
    if f["row_R"] < 0.6: obs.append("SPARSE_RIGHT")
    if f["row_R"] > 0.85: obs.append("FULL_RIGHT")
    if f["TR"] < f["TL"] * 0.4: obs.append("WEAK_TR")
    if f["BR"] < f["BL"] * 0.4: obs.append("WEAK_BR")
    if f["TR"] > f["TL"] * 1.5: obs.append("STRONG_TR")
    if f["BR"] > f["BL"] * 1.5: obs.append("STRONG_BR")
    if f["t_r_dens"] > 2 * f["b_r_dens"]: obs.append("TOP_HEAVY_R")
    if f["b_r_dens"] > 2 * f["t_r_dens"]: obs.append("BOT_HEAVY_R")
    
    obs_str = f"  [{', '.join(obs)}]" if obs else ""
    print(f"  POS {pos}: {top3}{obs_str}")
