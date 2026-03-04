#!/usr/bin/env python3
"""
Template-matching analysis for positions 6 and 11.

Strategy:
  1. Extract digit templates from the 10 KNOWN visible digits on the card
     (4,3,8,8,5,4 in prefix; 0,6,6,5 in suffix).
  2. For each hidden position (6 and 11), crop the zone and compare against
     every known template using:
       a) Normalised cross-correlation (cv2.matchTemplate TM_CCOEFF_NORMED)
       b) Mean squared error (MSE)
       c) Structural similarity via histogram correlation
       d) Horizontal/vertical pixel-profile similarity
  3. Aggregate scores across multiple enhancement pipelines.
  4. Also do an expanded OCR sweep with more enhancements specifically tuned
     for these two positions.

Geometry (from column brightness analysis):
  - ROI: y = [25%-55%] of image height
  - Pitch ≈ 75 px within groups; inter-group gap slightly wider
  - Known digit centers (in full-image x-coords):
      pos0='4': 197   pos1='3': 272   pos2='8': 347   pos3='8': 422
      pos4='5': 497   pos5='4': 572
      pos12='0':1133  pos13='6':1208  pos14='6':1283  pos15='5':1358
  - Hidden: pos6: 647   pos7: 722   pos8: 833   pos9: 908
            pos10: 983  pos11: 1058
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict

import cv2
import numpy as np
import pytesseract

IMG = "/Users/jenineferderas/Desktop/card_image.jpg"
WL = "-c tessedit_char_whitelist=0123456789"
OCR_EXCEPTIONS = (pytesseract.TesseractError, RuntimeError, TypeError, ValueError)
OCR_TIMEOUT_SEC = 1
FAST_OCR_SWEEP = True
MAX_OCR_CALLS_PER_POS = 2500 if FAST_OCR_SWEEP else 12000
OCR_SWEEP_DX = [-8, -3, 0, 3, 8] if FAST_OCR_SWEEP else [-12, -8, -5, -3, 0, 3, 5, 8, 12]
GRAY_ENH_LIMIT = 10 if FAST_OCR_SWEEP else None
BGR_ENH_LIMIT = 3 if FAST_OCR_SWEEP else 6

# =============================================================================
# Geometry
# =============================================================================
KNOWN_DIGITS: dict[int, str] = {
    0: "4", 1: "3", 2: "8", 3: "8", 4: "5", 5: "4",
    12: "0", 13: "6", 14: "6", 15: "5",
}

# X-centres of ALL 16 digit positions (estimated from column analysis + pitch)
CENTERS: dict[int, int] = {
    0: 197,  1: 272,  2: 347,  3: 422,
    4: 497,  5: 572,  6: 647,  7: 722,
    8: 833,  9: 908,  10: 983, 11: 1058,
    12: 1133, 13: 1208, 14: 1283, 15: 1358,
}

HALF = 34          # half-width of extraction window
SCALE = 4          # upscale factor
TARGET_POS = [6, 11]  # positions to analyse

# =============================================================================
# Image loading
# =============================================================================
img_bgr = cv2.imread(IMG)
if img_bgr is None:
    raise FileNotFoundError(f"Cannot load image: {IMG}")
h, w = img_bgr.shape[:2]
y0 = int(h * 0.25)
y1 = int(h * 0.55)
roi_bgr = img_bgr[y0:y1]
roi_gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
rh, rw = roi_gray.shape

print(f"Image: {w}x{h}, ROI: {rw}x{rh}")
print(f"ROI y-range: [{y0}, {y1}]\n")


# =============================================================================
# Enhancement pipelines
# =============================================================================
def make_enhancements(zone_gray: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """Return multiple enhanced versions of a digit zone."""
    variants: list[tuple[str, np.ndarray]] = []

    for clip in [4, 8, 16, 32, 64]:
        enh = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(3, 3)).apply(zone_gray)
        variants.extend(
            [
                (f"clahe{clip}_inv", cv2.bitwise_not(enh)),
                (f"clahe{clip}_raw", enh),
            ]
        )

    # Histogram eq inverted
    variants.append(("histeq_inv", cv2.bitwise_not(cv2.equalizeHist(zone_gray))))

    # Gamma corrections
    for gamma in [0.3, 0.5, 2.0]:
        lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], np.uint8)
        variants.append((f"gamma{gamma}_inv", cv2.bitwise_not(cv2.LUT(zone_gray, lut))))

    # Top-hat (extract bright features)
    tophat = cv2.morphologyEx(
        zone_gray,
        cv2.MORPH_TOPHAT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
    )
    variants.append(("tophat", tophat))

    # Morphological gradient
    grad = cv2.morphologyEx(zone_gray, cv2.MORPH_GRADIENT,
                            np.ones((3, 3), np.uint8))
    variants.append(("morph_grad", grad))

    # Sobel magnitude
    sobel_mag = np.hypot(
        cv2.Sobel(zone_gray, cv2.CV_64F, 1, 0, ksize=3),
        cv2.Sobel(zone_gray, cv2.CV_64F, 0, 1, ksize=3),
    )
    sobel_mag = (
        np.clip(sobel_mag / sobel_mag.max() * 255, 0, 255).astype(np.uint8)
        if sobel_mag.max() > 0
        else zone_gray
    )
    variants.append(("sobel_mag", sobel_mag))

    # Laplacian
    lap = cv2.Laplacian(zone_gray, cv2.CV_64F)
    lap = np.clip(np.abs(lap) / (np.abs(lap).max() + 1e-9) * 255, 0, 255).astype(np.uint8)
    variants.append(("laplacian", lap))

    # Unsharp mask
    usm = cv2.addWeighted(zone_gray, 2.0, cv2.GaussianBlur(zone_gray, (0, 0), 3), -1.0, 0)
    variants.append(("unsharp_inv", cv2.bitwise_not(usm)))

    return variants


def upscale(img: np.ndarray, factor: int = SCALE) -> np.ndarray:
    """Upscale for template comparison at higher resolution."""
    h2, w2 = img.shape[:2]
    return cv2.resize(img, (w2 * factor, h2 * factor), interpolation=cv2.INTER_CUBIC)


# =============================================================================
# Extract a digit zone (returns gray patch at SCALE resolution)
# =============================================================================
def extract_zone(position: int, x_shift: int = 0) -> np.ndarray:
    """Extract and upscale the zone for position `pos` with optional x-offset."""
    cx = CENTERS[position] + x_shift
    x0 = max(0, cx - HALF)
    x1 = min(rw, cx + HALF)
    zone = roi_gray[:, x0:x1]
    return upscale(zone)


def extract_zone_bgr(position: int, x_shift: int = 0) -> np.ndarray:
    """Extract BGR zone for a position."""
    cx = CENTERS[position] + x_shift
    x0 = max(0, cx - HALF)
    x1 = min(rw, cx + HALF)
    return roi_bgr[:, x0:x1]


# =============================================================================
# Template matching score
# =============================================================================
def ncc_score(template: np.ndarray, target: np.ndarray) -> float:
    """Normalised cross-correlation between two same-size images."""
    # resize to same size
    th, tw = template.shape[:2]
    tgt = cv2.resize(target, (tw, th), interpolation=cv2.INTER_CUBIC)
    t_f = template.astype(np.float64)
    g_f = tgt.astype(np.float64)
    t_f -= t_f.mean()
    g_f -= g_f.mean()
    denom = (np.linalg.norm(t_f) * np.linalg.norm(g_f))
    return 0.0 if denom < 1e-9 else float(np.sum(t_f * g_f) / denom)


def mse_score(template: np.ndarray, target: np.ndarray) -> float:
    """Mean-squared error (lower = more similar)."""
    th, tw = template.shape[:2]
    tgt = cv2.resize(target, (tw, th), interpolation=cv2.INTER_CUBIC)
    mse_diff = template.astype(np.float64) - tgt.astype(np.float64)
    return float(np.mean(mse_diff ** 2))


def profile_similarity(template: np.ndarray, target: np.ndarray) -> float:
    """Compare horizontal + vertical projection profiles via correlation."""
    th, tw = template.shape[:2]
    tgt = cv2.resize(target, (tw, th), interpolation=cv2.INTER_CUBIC)

    # Horizontal profile (mean per row)
    hp_t = template.astype(np.float64).mean(axis=1)
    hp_g = tgt.astype(np.float64).mean(axis=1)

    # Vertical profile (mean per column)
    vp_t = template.astype(np.float64).mean(axis=0)
    vp_g = tgt.astype(np.float64).mean(axis=0)

    def corr(a: np.ndarray, b: np.ndarray) -> float:
        a = a - a.mean()
        b = b - b.mean()
        corr_denom = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.sum(a * b) / corr_denom) if corr_denom > 1e-9 else 0.0

    return (corr(hp_t, hp_g) + corr(vp_t, vp_g)) / 2.0


# =============================================================================
# Build digit templates from known positions
# =============================================================================
print("=" * 60)
print("STEP 1: Build digit templates from known visible positions")
print("=" * 60)

# For each known digit, we store multiple enhanced versions
digit_templates: dict[str, list[np.ndarray]] = defaultdict(list)

for pos, digit in KNOWN_DIGITS.items():
    zone_gray_raw = extract_zone(pos)
    for _, enh_img in make_enhancements(zone_gray_raw):
        digit_templates[digit].append(enh_img)
    # Also BGR channels
    zone_bgr_raw = extract_zone_bgr(pos)
    zone_up = upscale(zone_bgr_raw)
    for ch_idx, _ in enumerate(["B", "G", "R"]):
        ch = zone_up[:, :, ch_idx]
        for _, enh_img in make_enhancements(ch)[:4]:  # top 4 for channels
            digit_templates[digit].append(enh_img)

# Deduplicate template digits
unique_template_digits = sorted(digit_templates.keys())
print(f"Template digits available: {unique_template_digits}")
for d in unique_template_digits:
    print(f"  '{d}': {len(digit_templates[d])} template variants")

# We also need templates for digits we DON'T have on the card: 1, 2, 7, 9
# We can't template-match these, but OCR can still suggest them
print("\nDigits WITHOUT templates on card: 1, 2, 7, 9")
print("These will rely on OCR evidence + structural inference\n")


# =============================================================================
# STEP 2: Template-match hidden positions against known templates
# =============================================================================
print("=" * 60)
print("STEP 2: Template matching for positions 6 and 11")
print("=" * 60)

for target_pos in TARGET_POS:
    print(f"\n--- POSITION {target_pos} (cx={CENTERS[target_pos]}) ---")

    # Aggregate NCC and profile scores per candidate digit
    ncc_totals: dict[str, list[float]] = defaultdict(list)
    prof_totals: dict[str, list[float]] = defaultdict(list)
    mse_totals: dict[str, list[float]] = defaultdict(list)

    # Try multiple x-offsets to handle small alignment errors
    for dx in [-8, -4, -2, 0, 2, 4, 8]:
        target_zone_raw = extract_zone(target_pos, dx)

        # Apply same enhancements to target
        for _, enh_target in make_enhancements(target_zone_raw):
            # Compare against each digit's templates (same enhancement)
            for digit, templates in digit_templates.items():
                best_ncc = -1.0
                best_prof = -1.0
                best_mse = 1e12

                # Compare against a subset of templates to keep it fast
                for tmpl in templates[:8]:
                    n = ncc_score(tmpl, enh_target)
                    p = profile_similarity(tmpl, enh_target)
                    m = mse_score(tmpl, enh_target)
                    best_ncc = max(best_ncc, n)
                    best_prof = max(best_prof, p)
                    best_mse = min(best_mse, m)

                ncc_totals[digit].append(best_ncc)
                prof_totals[digit].append(best_prof)
                mse_totals[digit].append(best_mse)

    # Print aggregated results
    print("\n  Template matching results (higher NCC/Profile = better match):\n")
    print(
        "  "
        f"{'Digit':>5}  {'AvgNCC':>7}  {'MaxNCC':>7}  "
        f"{'AvgProf':>8}  {'AvgMSE':>10}  {'Combined':>9}"
    )
    print(f"  {'─' * 5}  {'─' * 7}  {'─' * 7}  {'─' * 8}  {'─' * 10}  {'─' * 9}")

    combined: dict[str, float] = {}
    for digit in sorted(ncc_totals.keys()):
        avg_ncc = np.mean(ncc_totals[digit])
        max_ncc = np.max(ncc_totals[digit])
        avg_prof = np.mean(prof_totals[digit])
        avg_mse = np.mean(mse_totals[digit])

        # Combined score: weight NCC and profile equally, subtract normalised MSE
        # Higher is better
        comb = avg_ncc * 0.4 + max_ncc * 0.3 + avg_prof * 0.3
        combined[digit] = comb
        print(
            f"  '{digit}':  {avg_ncc:+.4f}  {max_ncc:+.4f}  "
            f"{avg_prof:+.4f}  {avg_mse:>10.0f}  {comb:+.4f}"
        )

    ranked = sorted(combined.items(), key=lambda x: -x[1])
    print(f"\n  >>> Template ranking for POS {target_pos}:")
    for i, (d, sc) in enumerate(ranked[:5], 1):
        print(f"      {i}. '{d}' (combined={sc:+.4f})")


# =============================================================================
# STEP 3: Expanded OCR sweep for positions 6 and 11
# =============================================================================
print("\n" + "=" * 60)
print("STEP 3: Deep OCR sweep for positions 6 and 11")
print("=" * 60)

OCR_CONFIGS = [
    f"--oem 3 --psm 10 {WL}",
    f"--oem 3 --psm 13 {WL}",
]
if not FAST_OCR_SWEEP:
    OCR_CONFIGS.extend([f"--oem 3 --psm 8 {WL}", f"--oem 3 --psm 7 {WL}"])
THRESHOLDS = [120, 150] if FAST_OCR_SWEEP else [100, 120, 130, 150, 170]


def _ocr_single(src: np.ndarray, cfg: str) -> str | None:
    """Run OCR on a single image and return the first digit found, or None."""
    try:
        txt = pytesseract.image_to_string(
            src,
            config=cfg,
            timeout=OCR_TIMEOUT_SEC,
        ).strip()
        digit_str = re.sub(r"\D", "", txt)
        if digit_str:
            return digit_str[0]
    except OCR_EXCEPTIONS:
        pass
    return None


def ocr_reads(img_in: np.ndarray, budget: int) -> tuple[list[str], int]:
    """Run OCR on one image, return (digit reads, OCR calls used)."""
    results: list[str] = []
    calls_used = 0
    if budget <= 0:
        return results, calls_used

    for cfg in OCR_CONFIGS:
        if calls_used >= budget:
            break
        if digit := _ocr_single(img_in, cfg):
            results.append(digit)
        calls_used += 1

        for thr in THRESHOLDS:
            if calls_used >= budget:
                break
            _, bw = cv2.threshold(img_in, thr, 255, cv2.THRESH_BINARY)
            if digit := _ocr_single(bw, cfg):
                results.append(digit)
            calls_used += 1
    return results, calls_used


for target_pos in TARGET_POS:
    print(f"\n--- OCR sweep: POSITION {target_pos} (cx={CENTERS[target_pos]}) ---")
    votes: Counter = Counter()
    budget_remaining = MAX_OCR_CALLS_PER_POS

    for dx in OCR_SWEEP_DX:
        if budget_remaining <= 0:
            break
        zone_raw = extract_zone(target_pos, dx)

        # Gray enhancements
        gray_enhs = make_enhancements(zone_raw)
        if GRAY_ENH_LIMIT is not None:
            gray_enhs = gray_enhs[:GRAY_ENH_LIMIT]
        for _, enh_img in gray_enhs:
            digit_reads, calls_used = ocr_reads(enh_img, budget_remaining)
            budget_remaining -= calls_used
            for digit_read in digit_reads:
                votes[digit_read] += 1
            if budget_remaining <= 0:
                break

        # BGR channels
        if budget_remaining <= 0:
            break
        zone_bgr = extract_zone_bgr(target_pos, dx)
        zone_bgr_up = upscale(zone_bgr)
        for ch_idx in range(3):
            if budget_remaining <= 0:
                break
            ch = zone_bgr_up[:, :, ch_idx]
            better_enhs = make_enhancements(ch)
            better_enhs = better_enhs[:BGR_ENH_LIMIT]
            for _, enh_img in better_enhs:
                digit_reads, calls_used = ocr_reads(enh_img, budget_remaining)
                budget_remaining -= calls_used
                for digit_read in digit_reads:
                    votes[digit_read] += 1
                if budget_remaining <= 0:
                    break

    total = sum(votes.values())
    calls_done = MAX_OCR_CALLS_PER_POS - budget_remaining
    print(f"  OCR calls used: {calls_done}/{MAX_OCR_CALLS_PER_POS}")
    if budget_remaining <= 0:
        print("  NOTE: OCR call budget reached; keeping highest-confidence votes.")
    print(f"  Total OCR reads: {total}")
    for digit, n in votes.most_common(8):
        pct = n / total * 100 if total else 0
        histogram = "#" * max(1, int(pct / 2))
        print(f"    '{digit}': {n:4d} ({pct:5.1f}%)  {histogram}")

    if votes:
        best = votes.most_common(1)[0]
        print(
            f"\n  >>> OCR verdict POS {target_pos} = '{best[0]}' "
            f"({best[1]}/{total} = {best[1] / total:.0%})"
        )


# =============================================================================
# STEP 4: Pixel-level structural analysis
# =============================================================================
print("\n" + "=" * 60)
print("STEP 4: Pixel structure analysis (edge density, symmetry)")
print("=" * 60)

for target_pos in TARGET_POS:
    zone_raw = extract_zone(target_pos)
    zone_enh = cv2.bitwise_not(cv2.createCLAHE(clipLimit=16.0, tileGridSize=(3, 3)).apply(zone_raw))

    # Edge map of target
    edges_target = cv2.Canny(zone_enh, 50, 150)
    edge_density_target = np.count_nonzero(edges_target) / edges_target.size

    # Vertical symmetry ratio
    half_w = zone_enh.shape[1] // 2
    left = zone_enh[:, :half_w]
    right = cv2.flip(zone_enh[:, -half_w:], 1)
    sym_score = ncc_score(left, right)

    # Horizontal projection profile shape
    h_profile = zone_enh.astype(np.float64).mean(axis=1)
    # Number of peaks suggests digit identity:
    # 0,6,8,9 → 2 horizontal peaks (top+bottom loops)
    # 1,4,7 → top-heavy or single-peak
    h_smooth = cv2.GaussianBlur(h_profile.reshape(-1, 1), (15, 1), 0).flatten()
    h_diff = np.diff(h_smooth)
    sign_changes = np.sum(np.diff(np.sign(h_diff)) != 0)

    print(f"\n  POS {target_pos}: edge_density={edge_density_target:.3f}, "
          f"v_symmetry={sym_score:.3f}, h_profile_peaks~{sign_changes}")

    # Compare edge density against known digits
    print("  Edge density comparison vs known digits:")
    for kpos, kdigit in sorted(KNOWN_DIGITS.items()):
        kzone = extract_zone(kpos)
        kenh = cv2.bitwise_not(cv2.createCLAHE(clipLimit=16.0, tileGridSize=(3, 3)).apply(kzone))
        kedges = cv2.Canny(kenh, 50, 150)
        k_edge_dens = np.count_nonzero(kedges) / kedges.size
        k_half_w = kenh.shape[1] // 2
        k_left = kenh[:, :k_half_w]
        k_right = cv2.flip(kenh[:, -k_half_w:], 1)
        k_sym = ncc_score(k_left, k_right)
        diff = abs(edge_density_target - k_edge_dens)
        print(f"    pos{kpos:2d}='{kdigit}': edge={k_edge_dens:.3f} "
              f"(Δ={diff:.3f}), sym={k_sym:.3f}")


# =============================================================================
# STEP 5: Save debug images
# =============================================================================
print("\n" + "=" * 60)
print("STEP 5: Saving debug comparison images")
print("=" * 60)

for target_pos in TARGET_POS:
    zone_raw = extract_zone(target_pos)
    # Best enhancement
    clahe16 = cv2.bitwise_not(
        cv2.createCLAHE(clipLimit=16.0, tileGridSize=(3, 3)).apply(zone_raw)
    )
    cv2.imwrite(f"/tmp/pos{target_pos}_clahe16_inv.png", clahe16)

    # Sobel
    sx = cv2.Sobel(zone_raw, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(zone_raw, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(sx ** 2 + sy ** 2)
    if mag.max() > 0:
        mag = (mag / mag.max() * 255).astype(np.uint8)
    cv2.imwrite(f"/tmp/pos{target_pos}_sobel.png", mag)

    # Save raw zone for visual comparison
    cv2.imwrite(f"/tmp/pos{target_pos}_raw_4x.png", zone_raw)

    # Side-by-side with all known templates (best enhancement)
    panels = []
    for kpos in sorted(KNOWN_DIGITS.keys()):
        kzone = extract_zone(kpos)
        kenh = cv2.bitwise_not(
            cv2.createCLAHE(clipLimit=16.0, tileGridSize=(3, 3)).apply(kzone)
        )
        # Label
        labeled = kenh.copy()
        digit_label = KNOWN_DIGITS[kpos]
        cv2.putText(labeled, f"p{kpos}={digit_label}", (5, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128.0, 128.0, 128.0), 2)
        panels.append(labeled)

    # Add the target
    target_labeled = clahe16.copy()
    cv2.putText(target_labeled, f"p{target_pos}=?", (5, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128.0, 128.0, 128.0), 2)
    panels.append(target_labeled)

    # Ensure all same height
    max_h = max(p.shape[0] for p in panels)
    max_w = max(p.shape[1] for p in panels)
    padded = []
    for p in panels:
        ph, pw = p.shape[:2]
        canvas = np.zeros((max_h, max_w), dtype=np.uint8)
        canvas[:ph, :pw] = p
        padded.append(canvas)

    comparison = np.hstack(padded)
    cv2.imwrite(f"/tmp/pos{target_pos}_comparison_strip.png", comparison)
    print(f"  Saved: /tmp/pos{target_pos}_comparison_strip.png")
    print(f"  Saved: /tmp/pos{target_pos}_clahe16_inv.png")
    print(f"  Saved: /tmp/pos{target_pos}_sobel.png")
    print(f"  Saved: /tmp/pos{target_pos}_raw_4x.png")


print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)
