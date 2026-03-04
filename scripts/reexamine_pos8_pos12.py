#!/usr/bin/env python3
"""
Re-examine PAN positions 8 and 12 with deep pixel analysis.

PAN layout (0-indexed):
  0123 4567 89AB CDEF
  4388 54?? ???? 0665
         ^       ^
        pos8    pos12

Position 8  = 3rd hidden digit (group 3, 1st char)
Position 12 = 1st digit of last group (assumed '0' — verify!)

Strategy:
  1. Column intensity profile to find actual digit x-positions
  2. Per-digit extraction at multiple scales + enhancements
  3. Vote consolidation
"""

import contextlib
import dataclasses
import itertools
import re
import sys
from collections import Counter
from typing import TypedDict

import cv2
import numpy as np
import pytesseract

IMG = sys.argv[1] if len(sys.argv) > 1 else "/Users/jenineferderas/Desktop/card_image.jpg"

OCR_WL = "-c tessedit_char_whitelist=0123456789"
OCR_EXCEPTIONS = (pytesseract.TesseractError, RuntimeError, TypeError, ValueError)


class DigitBox(TypedDict):
    """Bounding box and character for a detected digit."""

    ch: str
    x1: int
    y1: int
    x2: int
    y2: int


@dataclasses.dataclass
class ScanGeom:
    """Geometric parameters for digit scanning."""

    x_0665_start: float
    pitch: float
    half: float
    digit_y0: int
    digit_y1: int
    rw: int


@dataclasses.dataclass(frozen=True)
class PositionScanPlan:
    """Configuration for scanning a specific PAN position."""

    pos: int
    gap_factors: tuple[float, ...]
    shift_gap_factors: tuple[float, ...]
    dx_shifts: tuple[int, ...]
    log_channel_votes: bool = False
    log_gap_summary: bool = False


def _build_variants(gray_zone, up_fn):
    """Build enhancement variants for OCR."""
    variants = []

    for clip in [4, 8, 16, 32, 64]:
        c = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(3, 3))
        enh = c.apply(gray_zone)
        variants.extend([
            (f"clahe{clip}-inv", up_fn(cv2.bitwise_not(enh))),
            (f"clahe{clip}", up_fn(enh)),
        ])

    for k in [5, 7, 11]:
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        th = cv2.morphologyEx(gray_zone, cv2.MORPH_TOPHAT, kern)
        variants.append((f"tophat{k}", up_fn(th)))

    kernel = np.ones((3, 3), np.uint8)
    variants.append(("morphgrad", up_fn(cv2.morphologyEx(gray_zone, cv2.MORPH_GRADIENT, kernel))))

    he = cv2.equalizeHist(gray_zone)
    variants.extend([
        ("histeq-inv", up_fn(cv2.bitwise_not(he))),
        ("histeq", up_fn(he)),
    ])

    for gamma in [0.3, 0.5, 2.0]:
        lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype(np.uint8)
        g = cv2.LUT(gray_zone, lut)
        variants.extend([
            (f"gamma{gamma}-inv", up_fn(cv2.bitwise_not(g))),
            (f"gamma{gamma}", up_fn(g)),
        ])

    blur = cv2.GaussianBlur(gray_zone, (0, 0), 3)
    usm = cv2.addWeighted(gray_zone, 2.5, blur, -1.5, 0)
    variants.append(("usm-inv", up_fn(cv2.bitwise_not(usm))))

    sx = cv2.Sobel(gray_zone, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray_zone, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(sx ** 2 + sy ** 2)
    mag = np.clip(mag / mag.max() * 255, 0, 255).astype(np.uint8) if mag.max() > 0 else gray_zone
    variants.append(("sobel", up_fn(mag)))

    return variants


def _ocr_single_read(src, cfg):
    """Run OCR on a single image and return the first digit, or None."""
    with contextlib.suppress(*OCR_EXCEPTIONS):
        txt = pytesseract.image_to_string(src, config=cfg).strip()
        if d := re.sub(r"\D", "", txt):
            return d[0]
    return None


def ocr_zone(gray_zone, scale=8):
    """OCR a single-digit zone with multiple enhancements. Returns Counter of digits."""
    votes = Counter()
    h, w = gray_zone.shape[:2]

    def up(img):
        return cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)

    variants = _build_variants(gray_zone, up)

    configs = [
        f"--oem 3 --psm 10 {OCR_WL}",
        f"--oem 3 --psm 8 {OCR_WL}",
        f"--oem 3 --psm 13 {OCR_WL}",
        f"--oem 3 --psm 7 {OCR_WL}",
    ]

    thresholds = [0, 80, 100, 120, 140, 160, 180]

    for _vname, var_img in variants:
        for cfg in configs:
            for t in thresholds:
                src = var_img if t == 0 else cv2.threshold(var_img, t, 255, cv2.THRESH_BINARY)[1]
                digit = _ocr_single_read(src, cfg)
                if digit:
                    votes[digit] += 1

    return votes


def column_brightness(gray_row, smooth=5):
    """Compute per-column mean brightness with smoothing."""
    profile = gray_row.mean(axis=0)
    if smooth > 1:
        kernel = np.ones(smooth) / smooth
        profile = np.convolve(profile, kernel, mode="same")
    return profile


def find_digit_centers(profile, min_bright=20, min_gap=15):
    """Find digit center x-positions from brightness peaks in a row."""
    above = profile > min_bright
    centers = []
    in_peak = False
    start = 0
    for i, v in enumerate(above):
        if v and not in_peak:
            start = i
            in_peak = True
        elif not v and in_peak:
            c = (start + i) // 2
            if not centers or (c - centers[-1]) >= min_gap:
                centers.append(c)
            in_peak = False
    if in_peak:
        c = (start + len(above)) // 2
        if not centers or (c - centers[-1]) >= min_gap:
            centers.append(c)
    return centers


def _parse_charboxes(data: str, rh: int) -> list[DigitBox]:
    """Parse tesseract box output into DigitBox entries."""
    boxes: list[DigitBox] = []
    for line in data.strip().split("\n"):
        parts = line.split()
        if len(parts) >= 5:
            boxes.append({
                "ch": parts[0],
                "x1": int(parts[1]), "y1": rh - int(parts[4]),
                "x2": int(parts[3]), "y2": rh - int(parts[2]),
            })
    return boxes


def _evaluate_charbox_match(
    boxes: list[DigitBox], label: str,
    best: tuple[list[DigitBox], int, str, int],
) -> tuple[list[DigitBox], int, str, int]:
    """Check if '0665' appears in digit boxes, update best if better."""
    digit_boxes = [b for b in boxes if b["ch"].isdigit()]
    dtext = "".join(b["ch"] for b in digit_boxes)
    if "0665" not in dtext:
        return best
    print(f"  Found '0665' with {label}: digits='{dtext}'")
    idx = dtext.rfind("0665")
    for off in range(idx, idx + 4):
        b = digit_boxes[off]
        print(f"    '{b['ch']}' at x=[{b['x1']},{b['x2']}] y=[{b['y1']},{b['y2']}]")
    return (digit_boxes, idx, label, len(dtext)) if len(dtext) > best[3] else best


def _locate_suffix_via_charboxes(
    enh: np.ndarray, rh: int,
) -> tuple[list[DigitBox], int, str, int]:
    """Try to locate '0665' using character boxes. Returns (boxes, idx, label, dtext_len)."""
    best: tuple[list[DigitBox], int, str, int] = ([], -1, "N/A", -1)

    for psm in [6, 7, 11]:
        for src, label in [
            (cv2.bitwise_not(enh), f"clahe-inv/psm{psm}"),
            (enh, f"clahe/psm{psm}"),
        ]:
            try:
                data = pytesseract.image_to_boxes(src, config=f"--oem 3 --psm {psm}")
            except OCR_EXCEPTIONS:
                continue
            best = _evaluate_charbox_match(_parse_charboxes(data, rh), label, best)

    return best


def _resolve_geometry(
    best_digit_boxes: list[DigitBox], best_idx_0665: int, best_label: str,
    centers: list[int], rh: int, rw: int,
) -> tuple[float, float, int, int]:
    """Determine (x_0665_start, pitch, digit_y0, digit_y1) from charboxes or fallback."""
    if not best_digit_boxes or best_idx_0665 < 0:
        print("  WARNING: Could not locate '0665' via char boxes!")
        print("  Falling back to column-profile estimation.")
        if len(centers) >= 4:
            suffix_centers = centers[-4:]
            pitch = (suffix_centers[-1] - suffix_centers[0]) / 3.0
        else:
            pitch = 75.0
            suffix_centers = [rw - 290, rw - 215, rw - 140, rw - 65]
        x_0665_start = float(suffix_centers[0])
        digit_y0, digit_y1 = int(rh * 0.15), int(rh * 0.85)
        print(f"  Estimated pitch: {pitch:.1f}px")
        print(f"  Estimated '0' of 0665 at x={x_0665_start}")
    else:
        print(f"\n  Using best match: {best_label}")
        idx = best_idx_0665
        b0 = best_digit_boxes[idx]
        b5 = best_digit_boxes[idx + 3]
        pitch = (b5["x1"] - b0["x1"]) / 3.0
        x_0665_start = float(b0["x1"])
        dh = b0["y2"] - b0["y1"]
        digit_y0 = max(0, b0["y1"] - int(dh * 0.3))
        digit_y1 = min(rh, b0["y2"] + int(dh * 0.3))
        print(f"  Pitch from suffix: {pitch:.1f}px")
        print(f"  '0' of '0665' at x={b0['x1']}, digit height={dh}")
    return x_0665_start, pitch, digit_y0, digit_y1


def _print_votes(label: str, votes: Counter, top_n: int = 8) -> None:
    """Print vote distribution for a position."""
    total = sum(votes.values())
    print(f"\n  {label} — Total reads: {total}")
    for d, n in votes.most_common(top_n):
        pct = n / total * 100 if total else 0
        histogram = "█" * max(1, int(pct / 2))
        print(f"    '{d}': {n:5d} ({pct:5.1f}%)  {histogram}")


def _print_section(title: str, note: str | None = None) -> None:
    """Print a standardized section header."""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    if note:
        print(f"  {note}\n")


def _save_debug_images(
    gray: np.ndarray, digit_y0: int, digit_y1: int,
    zx0: int, zx1: int, prefix: str,
) -> None:
    """Save raw, enhanced, and top-hat debug images for a digit zone."""
    zone = gray[digit_y0:digit_y1, zx0:zx1]
    cv2.imwrite(f"/tmp/{prefix}_raw.png", zone)
    c = cv2.createCLAHE(clipLimit=32.0, tileGridSize=(3, 3))
    enh = cv2.bitwise_not(c.apply(zone))
    big = cv2.resize(enh, (enh.shape[1] * 10, enh.shape[0] * 10), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(f"/tmp/{prefix}_enhanced.png", big)
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    th = cv2.morphologyEx(zone, cv2.MORPH_TOPHAT, kern)
    th_big = cv2.resize(th, (th.shape[1] * 10, th.shape[0] * 10), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(f"/tmp/{prefix}_tophat.png", th_big)
    print(f"  Debug: /tmp/{prefix}_raw.png, /tmp/{prefix}_enhanced.png, /tmp/{prefix}_tophat.png")


def _center_for_position(geom: ScanGeom, pos: int, gap_factor: float = 0.5) -> float:
    """Compute x-center for a PAN position using suffix anchor and pitch."""
    if pos == 12:
        return geom.x_0665_start
    steps_left = 12 - pos
    return geom.x_0665_start - (steps_left - 0.5 + gap_factor) * geom.pitch


def _scan_channel_pass(
    channels: dict[str, np.ndarray], geom: ScanGeom, plan: PositionScanPlan,
) -> Counter:
    """Run channel-based OCR pass for all configured gap factors."""
    votes = Counter()
    for gap_factor in plan.gap_factors:
        x_center = _center_for_position(geom, plan.pos, gap_factor)
        zx0 = max(0, int(x_center - geom.half))
        zx1 = min(geom.rw, int(x_center + geom.half))

        gap_votes = Counter()
        for ch_name, ch_img in channels.items():
            zone = ch_img[geom.digit_y0:geom.digit_y1, zx0:zx1]
            if zone.size == 0:
                continue
            channel_votes = ocr_zone(zone, scale=8)
            gap_votes += channel_votes
            if plan.log_channel_votes and channel_votes:
                print(f"  Channel '{ch_name}': {channel_votes.most_common(5)}")

        if plan.log_gap_summary and gap_votes:
            top3 = gap_votes.most_common(3)
            total = sum(gap_votes.values())
            info = ", ".join(f"'{d}'={n / total:.0%}" for d, n in top3)
            print(f"  gap={gap_factor:.1f}: center_x={x_center:.0f}, zone=[{zx0},{zx1}] → {info}")

        votes += gap_votes
    return votes


def _scan_shift_pass(gray: np.ndarray, geom: ScanGeom, plan: PositionScanPlan) -> Counter:
    """Run shifted gray OCR pass for robust position refinement."""
    votes = Counter()
    for gap_factor in plan.shift_gap_factors:
        x_center = _center_for_position(geom, plan.pos, gap_factor)
        for dx in plan.dx_shifts:
            zx0 = max(0, int(x_center - geom.half + dx))
            zx1 = min(geom.rw, int(x_center + geom.half + dx))
            zone = gray[geom.digit_y0:geom.digit_y1, zx0:zx1]
            if zone.size == 0:
                continue
            votes += ocr_zone(zone, scale=8)
    return votes


def _scan_position(
    channels: dict[str, np.ndarray], gray: np.ndarray, geom: ScanGeom, plan: PositionScanPlan,
) -> Counter:
    """Scan one position using channel pass plus shifted gray pass."""
    votes = _scan_channel_pass(channels, geom, plan)
    votes += _scan_shift_pass(gray, geom, plan)
    return votes


def _luhn_valid(number: str) -> bool:
    """Return True when number satisfies the Luhn checksum."""
    if not number.isdigit() or len(number) != 16:
        return False
    total = 0
    for i, ch in enumerate(reversed(number)):
        d = int(ch)
        if i % 2 == 1:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
    return total % 10 == 0


def _top_digits(votes: Counter, top_k: int = 3) -> list[tuple[str, float]]:
    """Return top-k digits with normalized vote probabilities."""
    total = sum(votes.values())
    if total == 0:
        return [(str(d), 0.1) for d in range(10)]
    return [(digit, count / total) for digit, count in votes.most_common(top_k)]


def _print_luhn_recompute(votes_by_pos: dict[int, Counter]) -> None:
    """Recompute and rank Luhn-valid PAN candidates from updated OCR evidence."""
    print("\n" + "=" * 60)
    print("=== LUHN RECOMPUTE with updated evidence ===")
    print("=" * 60)

    pos_order = [8, 9, 10, 11, 12]
    digit_options = [_top_digits(votes_by_pos[pos], top_k=3) for pos in pos_order]

    candidates: list[tuple[str, float]] = []
    for combo in itertools.product(*digit_options):
        digits = [d for d, _p in combo]
        score = float(np.prod([p for _d, p in combo]))
        pan_chars = list("43885454?????665")
        for pos, digit in zip(pos_order, digits):
            pan_chars[pos] = digit
        pan = "".join(pan_chars)
        if _luhn_valid(pan):
            candidates.append((pan, score))

    candidates.sort(key=lambda x: x[1], reverse=True)

    combos = 1
    for opts in digit_options:
        combos *= len(opts)
    print(f"  Candidate combinations checked: {combos}")
    print(f"  Luhn-valid candidates: {len(candidates)}")
    for rank, (pan, score) in enumerate(candidates[:10], start=1):
        grouped = f"{pan[:4]} {pan[4:8]} {pan[8:12]} {pan[12:]}"
        print(f"  #{rank:2d}  {grouped}  score={score:.6f}")


def main():
    """Run deep OCR re-examination for PAN positions 8, 9, 10, 11, and 12."""
    img = cv2.imread(IMG)
    if img is None:
        sys.exit(f"Cannot load {IMG}")
    h, w = img.shape[:2]
    print(f"Image: {w}x{h}\n")

    y0, y1 = int(h * 0.30), int(h * 0.62)
    row = img[y0:y1]
    gray = cv2.cvtColor(row, cv2.COLOR_BGR2GRAY)
    rh, rw = gray.shape

    clahe = cv2.createCLAHE(clipLimit=16.0, tileGridSize=(4, 4))
    enh = clahe.apply(gray)

    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kern)

    print("=== Column brightness analysis (top-hat, smoothed) ===")
    profile = column_brightness(tophat, smooth=7)
    centers = find_digit_centers(profile, min_bright=8, min_gap=20)
    print(f"  Detected {len(centers)} potential digit centers: {centers}")

    if len(centers) >= 4:
        gaps = [centers[i + 1] - centers[i] for i in range(len(centers) - 1)]
        median_gap = sorted(gaps)[len(gaps) // 2] if gaps else 50
        print(f"  Inter-center gaps: {gaps}")
        print(f"  Median gap: {median_gap}")

    print("\n=== Char-box location of suffix '0665' ===")
    best_digit_boxes, best_idx_0665, best_label, _ = _locate_suffix_via_charboxes(enh, rh)

    x_0665_start, pitch, digit_y0, digit_y1 = _resolve_geometry(
        best_digit_boxes, best_idx_0665, best_label, centers, rh, rw,
    )
    print(f"\n  Digit row y-range in ROI: [{digit_y0}, {digit_y1}]")

    geom = ScanGeom(
        x_0665_start=x_0665_start,
        pitch=pitch,
        half=pitch * 0.6,
        digit_y0=digit_y0,
        digit_y1=digit_y1,
        rw=rw,
    )
    blue, green, red = row[:, :, 0], row[:, :, 1], row[:, :, 2]
    channels = {"gray": gray, "blue": blue, "green": green, "red": red}

    print("\n" + "=" * 60)
    print("=== POSITION 12 — Deep re-examination ===")
    print("=" * 60)
    print("  (We assumed this is '0'. Let's verify.)\n")

    pos12_votes = _scan_position(
        channels,
        gray,
        geom,
        PositionScanPlan(
            pos=12,
            gap_factors=(0.0,),
            shift_gap_factors=(0.0,),
            dx_shifts=(-10, -5, 5, 10),
            log_channel_votes=True,
        ),
    )
    _print_votes("POS 12", pos12_votes)

    x_c12 = _center_for_position(geom, pos=12, gap_factor=0.0)
    zx0 = max(0, int(x_c12 - geom.half))
    zx1 = min(geom.rw, int(x_c12 + geom.half))
    _save_debug_images(gray, geom.digit_y0, geom.digit_y1, zx0, zx1, "pos12")

    print("\n" + "=" * 60)
    print("=== POSITION 8 — Deep re-examination ===")
    print("=" * 60)
    print("  (Previous evidence: '5' at 48%, '8' at 41%)\n")

    pos8_votes = _scan_position(
        channels,
        gray,
        geom,
        PositionScanPlan(
            pos=8,
            gap_factors=(0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0),
            shift_gap_factors=(0.3, 0.5),
            dx_shifts=(-15, -8, 8, 15),
            log_gap_summary=True,
        ),
    )
    _print_votes("POS 8", pos8_votes)

    x_c8 = _center_for_position(geom, pos=8, gap_factor=0.4)
    zx0 = max(0, int(x_c8 - geom.half))
    zx1 = min(geom.rw, int(x_c8 + geom.half))
    _save_debug_images(gray, geom.digit_y0, geom.digit_y1, zx0, zx1, "pos8")

    pos_votes: dict[int, Counter] = {8: pos8_votes, 12: pos12_votes}
    for pos in [9, 10, 11]:
        print("\n" + "=" * 60)
        print(f"=== POSITION {pos} — Deep re-examination ===")
        print("=" * 60)

        votes = _scan_position(
            channels,
            gray,
            geom,
            PositionScanPlan(
                pos=pos,
                gap_factors=(0.4, 0.5, 0.6),
                shift_gap_factors=(0.5,),
                dx_shifts=(-10, -5, 5, 10),
            ),
        )
        pos_votes[pos] = votes
        _print_votes(f"POS {pos}", votes)

        x_center = _center_for_position(geom, pos=pos, gap_factor=0.5)
        zx0 = max(0, int(x_center - geom.half))
        zx1 = min(geom.rw, int(x_center + geom.half))
        _save_debug_images(gray, geom.digit_y0, geom.digit_y1, zx0, zx1, f"pos{pos}")

    print("\n" + "=" * 60)
    print("=== SUMMARY ===")
    print("=" * 60)
    print("  PAN template: 4388 54?? ???? 0665")
    print(f"  Pitch: {pitch:.1f}px")

    for pos in [8, 9, 10, 11, 12]:
        label = f"POS {pos}"
        votes = pos_votes[pos]
        total = sum(votes.values())
        top5 = votes.most_common(5)
        info = ", ".join(f"'{d}'={n / total:.1%}" for d, n in top5) if total else "(no votes)"
        print(f"  {label}: {info}  (n={total})")

    _print_luhn_recompute(pos_votes)


if __name__ == "__main__":
    main()
