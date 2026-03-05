"""Lean re-examination of PAN positions 8 and 12. Minimal OCR calls."""

import re
import sys
from itertools import product
from collections import Counter

import cv2
import numpy as np
import pytesseract

IMG = sys.argv[1] if len(sys.argv) > 1 else "/Users/jenineferderas/Desktop/card_image.jpg"
OCR_WL = "-c tessedit_char_whitelist=0123456789"
OCR_EXCEPTIONS = (pytesseract.TesseractError, RuntimeError, TypeError, ValueError)


def _build_variants(gray_zone: np.ndarray, scale: int) -> list[np.ndarray]:
    """Build a small set of enhancement variants for one digit zone."""
    h, w = gray_zone.shape[:2]
    variants: list[np.ndarray] = []

    for clip in [8, 32]:
        c = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(3, 3))
        enh = c.apply(gray_zone)
        up_inv = cv2.resize(cv2.bitwise_not(enh), (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
        up_raw = cv2.resize(enh, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
        variants.extend([up_inv, up_raw])

    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    th = cv2.morphologyEx(gray_zone, cv2.MORPH_TOPHAT, kern)
    variants.append(cv2.resize(th, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC))

    lut = np.array([((i / 255.0) ** 0.3) * 255 for i in range(256)]).astype(np.uint8)
    gam = cv2.LUT(gray_zone, lut)
    variants.append(cv2.resize(cv2.bitwise_not(gam), (w * scale, h * scale), interpolation=cv2.INTER_CUBIC))

    return variants


def _ocr_first_digit(src: np.ndarray, cfg: str) -> str | None:
    """Run OCR and return the first detected digit, if any."""
    try:
        txt = pytesseract.image_to_string(src, config=cfg).strip()
    except OCR_EXCEPTIONS:
        return None
    return d[0] if (d := re.sub(r"\D", "", txt)) else None


def ocr_zone_fast(gray_zone: np.ndarray, scale: int = 6) -> Counter:
    """OCR a single-digit zone — lean variant."""
    votes: Counter = Counter()
    variants = _build_variants(gray_zone, scale)
    configs = [
        f"--oem 3 --psm 10 {OCR_WL}",
        f"--oem 3 --psm 13 {OCR_WL}",
    ]
    thresholds = [0, 120, 160]

    for var_img in variants:
        for cfg, thresh in product(configs, thresholds):
            src = var_img if thresh == 0 else cv2.threshold(var_img, thresh, 255, cv2.THRESH_BINARY)[1]
            if digit := _ocr_first_digit(src, cfg):
                votes[digit] += 1
    return votes


def _parse_digit_boxes(data: str, rh: int) -> list[dict[str, int | str]]:
    """Parse tesseract box output into digit box dicts."""
    boxes: list[dict[str, int | str]] = []
    for line in data.strip().split("\n"):
        parts = line.split()
        if len(parts) < 5 or not parts[0].isdigit():
            continue
        x1, y1b, x2, y2b = map(int, parts[1:5])
        boxes.append({"ch": parts[0], "x1": x1, "y1": rh - y2b, "x2": x2, "y2": rh - y1b})
    return boxes


def _suffix_from_boxes(
    boxes: list[dict[str, int | str]], rh: int,
) -> tuple[float, float, int, int] | None:
    """Extract suffix geometry from digit boxes if '0665' is found."""
    dtext = "".join(str(b["ch"]) for b in boxes)
    idx = dtext.rfind("0665")
    if idx < 0:
        return None
    b0 = boxes[idx]
    b5 = boxes[idx + 3]
    x0 = int(b0["x1"])
    pitch = (int(b5["x1"]) - x0) / 3.0
    dy = int(b0["y2"]) - int(b0["y1"])
    y0 = max(0, int(b0["y1"]) - int(dy * 0.3))
    y1 = min(rh, int(b0["y2"]) + int(dy * 0.3))
    return float(x0), pitch, y0, y1


def _find_suffix_via_boxes(
    enh: np.ndarray, rh: int,
) -> tuple[float, float, int, int] | None:
    """Try to locate '0665' using character boxes. Returns (suffix_x, pitch, y0, y1) or None."""
    for psm, src in product([6, 7, 11], [cv2.bitwise_not(enh), enh]):
        try:
            data = pytesseract.image_to_boxes(src, config=f"--oem 3 --psm {psm}")
        except OCR_EXCEPTIONS:
            continue
        if result := _suffix_from_boxes(_parse_digit_boxes(data, rh), rh):
            suffix_x, pitch, y0, y1 = result
            print(f"Found '0665' via psm{psm}: x={suffix_x}, pitch={pitch:.1f}")
            return suffix_x, pitch, y0, y1
    return None


def _find_suffix_via_profile(gray: np.ndarray, rw: int) -> tuple[float, float]:
    """Fallback suffix location from column brightness profile. Returns (suffix_x, pitch)."""
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kern)
    profile = tophat.mean(axis=0)
    k = np.ones(7) / 7
    profile = np.convolve(profile, k, mode="same")
    above = profile > 8
    centers: list[int] = []
    in_pk = False
    start = 0
    for i, v in enumerate(above):
        if v and not in_pk:
            start = i
            in_pk = True
        elif not v and in_pk:
            c = (start + i) // 2
            if not centers or c - centers[-1] >= 20:
                centers.append(c)
            in_pk = False
    if len(centers) >= 4:
        suffix_centers = centers[-4:]
        pitch = (suffix_centers[-1] - suffix_centers[0]) / 3.0
        return float(suffix_centers[0]), pitch
    return float(rw - 290), 75.0


def _print_votes(votes: Counter, top_n: int = 6) -> None:
    """Print vote distribution."""
    total = sum(votes.values())
    print(f"Total reads: {total}")
    for d, n in votes.most_common(top_n):
        pct = n / total * 100 if total else 0
        bar = "█" * max(1, int(pct / 2))
        print(f"  '{d}': {n:4d} ({pct:5.1f}%)  {bar}")


def _print_section(title: str) -> None:
    """Print a section header."""
    print("\n" + "=" * 50)
    print(title)
    print("=" * 50)


def _scan_position(
    channels: list[np.ndarray],
    suffix_x: float, pitch: float, half: float,
    digit_y0_r: int, digit_y1_r: int, rw: int,
    pos_offset: float, gap_factors: list[float], dx_values: list[int],
) -> Counter:
    """Scan a single digit position across channels and offsets."""
    votes: Counter = Counter()
    for gap_factor in gap_factors:
        gap = pitch * gap_factor
        cx = suffix_x - gap - pos_offset * pitch
        for dx in dx_values:
            zx0 = max(0, int(cx + dx - half))
            zx1 = min(rw, int(cx + dx + half))
            for ch in channels:
                zone = ch[digit_y0_r:digit_y1_r, zx0:zx1]
                if zone.size == 0:
                    continue
                votes += ocr_zone_fast(zone)
    return votes


def main() -> None:
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

    digit_y0_r, digit_y1_r = int(rh * 0.15), int(rh * 0.85)
    result = _find_suffix_via_boxes(enh, rh)
    if result is not None:
        suffix_x, pitch, digit_y0_r, digit_y1_r = result
    else:
        suffix_x, pitch = _find_suffix_via_profile(gray, rw)
        print(f"Fallback: suffix_x={suffix_x}, pitch={pitch:.1f}")

    half = pitch * 0.6
    print(f"Pitch: {pitch:.1f}px, y-range: [{digit_y0_r}, {digit_y1_r}]")

    channels = [gray, row[:, :, 0], row[:, :, 1], row[:, :, 2]]

    _print_section("POSITION 12 (assumed '0' of '0665')")

    pos12_votes = _scan_position(
        channels, suffix_x, pitch, half, digit_y0_r, digit_y1_r, rw,
        pos_offset=0.0,
        gap_factors=[0.0],
        dx_values=[-10, -5, 0, 5, 10],
    )
    _print_votes(pos12_votes)

    _print_section("POSITION 8 (3rd hidden digit, 1st of group 3)")
    print("(Previous: '5'=48%, '8'=41%)\n")

    pos8_votes = _scan_position(
        channels, suffix_x, pitch, half, digit_y0_r, digit_y1_r, rw,
        pos_offset=3.5,
        gap_factors=[0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0],
        dx_values=[-8, 0, 8],
    )
    _print_votes(pos8_votes)

    for pos_name, pos_offset in [("POS 9 (4th hidden)", 2.5), ("POS 10 (5th hidden)", 1.5), ("POS 11 (6th hidden)", 0.5)]:
        _print_section(pos_name)
        votes = _scan_position(
            channels, suffix_x, pitch, half, digit_y0_r, digit_y1_r, rw,
            pos_offset=pos_offset,
            gap_factors=[0.0, 0.3, 0.5, 0.7, 1.0],
            dx_values=[-5, 0, 5],
        )
        _print_votes(votes)

    _print_section("LUHN RECOMPUTE with updated evidence")


if __name__ == "__main__":
    main()
