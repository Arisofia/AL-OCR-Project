#!/usr/bin/env python3
# pylint: disable=too-many-nested-blocks
"""
Deep zoom analysis of positions 10-11 (last two hidden digits before 0665).

Uses multiple zone offsets, finer enhancements, channel-separated analysis,
and saves debug images for visual inspection.
"""

import sys
import re
from collections import Counter

import cv2
import numpy as np
import pytesseract

# ── Image loading ──────────────────────────────────────────────────────────

def load(path: str) -> np.ndarray:
    """Load image from disk and abort if path cannot be opened."""
    img = cv2.imread(path)
    if img is None:
        sys.exit(f"ERROR: cannot load {path}")
    return img


# ── Marker detection (refined) ────────────────────────────────────────────

def find_marker(img: np.ndarray):
    """Locate the dark marker strip covering the hidden digits."""
    h, _ = img.shape[:2]
    y0, y1 = int(h * 0.25), int(h * 0.65)
    gray = cv2.cvtColor(img[y0:y1], cv2.COLOR_BGR2GRAY)
    col_mean = np.mean(gray, axis=0)
    thresh = np.percentile(col_mean, 25)
    dark = np.nonzero(col_mean < thresh)[0]
    x0, x1 = int(dark[0]), int(dark[-1])
    print(f"Marker x=[{x0}, {x1}], y=[{y0}, {y1}], width={x1-x0}")
    return x0, x1, y0, y1


# ── Enhancement battery ──────────────────────────────────────────────────

def make_variants(gray: np.ndarray, scale: int = 5) -> list:  # pylint: disable=too-many-locals
    """Return [(name, upscaled_gray), ...] with many enhancements."""
    h, w = gray.shape[:2]
    out = []

    def up(img):
        return cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)

    # CLAHE sweep
    for clip in [2, 4, 8, 16, 32, 64]:
        c = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(3, 3))
        out.append((f"clahe{clip}", up(cv2.bitwise_not(c.apply(gray)))))

    # Morph gradient
    for k in [2, 3, 5]:
        kernel = np.ones((k, k), np.uint8)
        g = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
        out.append((f"morphgrad{k}", up(g)))

    # Sobel
    sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(sx**2 + sy**2)
    mag = np.clip(255 * mag / (mag.max() + 1e-6), 0, 255).astype(np.uint8)
    out.append(("sobel", up(mag)))

    # Laplacian
    lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
    lap = np.clip(np.abs(lap), 0, 255).astype(np.uint8)
    out.append(("laplacian", up(lap)))

    # Histogram equalization
    eq = cv2.equalizeHist(gray)
    out.append(("histeq", up(cv2.bitwise_not(eq))))

    # Unsharp masks with varying strength
    for sigma, alpha in [(3, 2.0), (5, 3.0), (7, 4.0)]:
        blurred = cv2.GaussianBlur(gray, (0, 0), sigma)
        us = cv2.addWeighted(gray, alpha, blurred, -(alpha - 1), 0)
        out.append((f"unsharp-s{sigma}-a{alpha}", up(cv2.bitwise_not(us))))

    # Top-hat (reveal bright detail on dark background)
    for k in [5, 9, 15]:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        th = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        out.append((f"tophat{k}", up(th)))

    # Black-hat (dark detail on lighter background)
    for k in [5, 9, 15]:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        out.append((f"blackhat{k}", up(bh)))

    # Gamma corrections
    for gamma in [0.3, 0.5, 2.0, 3.0]:
        lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype(np.uint8)
        gc = cv2.LUT(gray, lut)
        out.append((f"gamma{gamma}", up(cv2.bitwise_not(gc))))

    return out


# ── OCR battery ──────────────────────────────────────────────────────────

OCR_CONFIGS = [
    ("psm7",  "--oem 3 --psm 7  -c tessedit_char_whitelist=0123456789"),
    ("psm8",  "--oem 3 --psm 8  -c tessedit_char_whitelist=0123456789"),
    ("psm10", "--oem 3 --psm 10 -c tessedit_char_whitelist=0123456789"),
    ("psm13", "--oem 3 --psm 13 -c tessedit_char_whitelist=0123456789"),
    ("psm6",  "--oem 3 --psm 6  -c tessedit_char_whitelist=0123456789"),
]

THRESHOLDS = [80, 100, 120, 140, 160, 180]
OCR_EXCEPTIONS = (pytesseract.TesseractError, RuntimeError, TypeError, ValueError)


def ocr_all(img_gray_upscaled: np.ndarray) -> list[tuple[str, str]]:  # NOSONAR
    """Run OCR configs + thresholds on a single variant image."""
    results = []
    for mode, cfg in OCR_CONFIGS:
        try:
            txt = pytesseract.image_to_string(img_gray_upscaled, config=cfg).strip()
            d = re.sub(r"\D", "", txt)
            if d:
                results.append((mode, d))
        except OCR_EXCEPTIONS:
            pass
        for t in THRESHOLDS:
            _, bw = cv2.threshold(img_gray_upscaled, t, 255, cv2.THRESH_BINARY)
            try:
                txt = pytesseract.image_to_string(bw, config=cfg).strip()
                d = re.sub(r"\D", "", txt)
                if d:
                    results.append((f"{mode}/t{t}", d))
            except OCR_EXCEPTIONS:
                pass
            # Also try inverted
            try:
                txt = pytesseract.image_to_string(255 - bw, config=cfg).strip()
                d = re.sub(r"\D", "", txt)
                if d:
                    results.append((f"{mode}/t{t}inv", d))
            except OCR_EXCEPTIONS:
                pass
    return results


# ── Per-position digit voting ────────────────────────────────────────────

def vote_digits(all_reads: list) -> Counter:
    """Count single-digit votes from OCR reads (take first char of each read)."""
    votes = Counter()
    for _, digits in all_reads:
        if len(digits) >= 1:
            votes[digits[0]] += 1
    return votes


def vote_all_chars(all_reads: list) -> Counter:
    """Count every character across all reads."""
    votes = Counter()
    for _, digits in all_reads:
        for ch in digits:
            votes[ch] += 1
    return votes


# ── Main ─────────────────────────────────────────────────────────────────

def main() -> None:  # pylint: disable=too-many-locals,too-many-branches,too-many-statements,too-many-nested-blocks  # NOSONAR
    """Run deep zoom analysis for PAN positions 10 and 11."""
    path = sys.argv[1] if len(sys.argv) > 1 else "/Users/jenineferderas/Desktop/card_image.jpg"
    img = load(path)
    h, w = img.shape[:2]
    print(f"Image {w}x{h}\n")

    mx0, mx1, my0, my1 = find_marker(img)
    marker_w = mx1 - mx0

    # ── Step 1: Try different digit-count assumptions ──
    # The marker might cover 6, 7, or 8 digits depending on spacing
    # Positions 10-11 are the 5th-6th hidden digit from left
    # With standard Visa formatting: 4388 54XX XXXX 0665
    # Hidden = positions 6-11 (0-indexed in the 16-digit PAN)
    # In the 4-group format: group2[2:4] + group3[0:4] = 6 hidden digits

    print("\n══════════════════════════════════════════════════════════")
    print("  DEEP ZOOM: Positions 10 & 11 (last 2 hidden before 0665)")
    print("══════════════════════════════════════════════════════════\n")

    # We know the marker covers 6 digits. The right edge of the marker
    # transitions into the visible "0665". Let me try multiple offsets
    # from the RIGHT edge of the marker working leftward.

    # Estimate single digit width from known visible text
    # "4388 54" = 6 digits + space(s) before marker ≈ mx0 pixels
    # "0665" = 4 digits after marker ≈ (w - mx1) pixels
    # each digit ≈ (w - mx1) / 4, but there are margins
    # Better: marker_w / 6 digits
    digit_est = marker_w / 6.0
    print(f"Estimated digit width: {digit_est:.1f}px (marker_w={marker_w}/6)")

    # Colour channels separated (embossing may show up differently in R/G/B)
    bgr = img[my0:my1, :]
    channels = {
        "gray": cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY),
        "blue": bgr[:, :, 0],
        "green": bgr[:, :, 1],
        "red": bgr[:, :, 2],
    }

    # ── For each of positions 10, 11: try multiple zone offsets ──
    for target_pos in [10, 11]:
        idx = target_pos - 6  # 0-based index in the 6 hidden digits (4 or 5)
        print(f"\n{'─'*60}")
        print(f"  POSITION {target_pos} (hidden digit #{idx+1} of 6)")
        print(f"{'─'*60}")

        grand_votes = Counter()

        # Try several zone-width and offset assumptions
        for dw_factor in [0.85, 0.90, 0.95, 1.0, 1.05, 1.10, 1.15]:
            dw = digit_est * dw_factor
            # Center of digit
            cx = mx0 + (idx + 0.5) * dw
            # Extract zone centered there
            pad = dw * 0.65  # some overlap
            zx0 = max(0, int(cx - pad))
            zx1 = min(w, int(cx + pad))

            for ch_img in channels.values():
                zone = ch_img[my0:my1, zx0:zx1] if ch_img.ndim == 2 else ch_img[:, zx0:zx1]
                if zone.size == 0:
                    continue
                variants = make_variants(zone, scale=5)
                for _, var_img in variants:
                    reads = ocr_all(var_img)
                    for _, digits in reads:
                        if len(digits) >= 1:
                            grand_votes[digits[0]] += 1

        # Also try from the RIGHT edge: position 11 is immediately left of suffix
        # and position 10 is one digit-width further left
        for shift_px in range(-15, 16, 5):
            # right_edge_of_digit_11 ≈ mx1 (end of marker)
            # So digit 11 center ≈ mx1 - digit_est/2
            # digit 10 center ≈ mx1 - digit_est*1.5
            if target_pos == 11:
                cx_right = mx1 - digit_est * 0.5 + shift_px
            else:
                cx_right = mx1 - digit_est * 1.5 + shift_px

            pad = digit_est * 0.65
            zx0 = max(0, int(cx_right - pad))
            zx1 = min(w, int(cx_right + pad))

            for ch_img in channels.values():
                zone = ch_img[my0:my1, zx0:zx1] if ch_img.ndim == 2 else ch_img[:, zx0:zx1]
                if zone.size == 0:
                    continue
                variants = make_variants(zone, scale=5)
                for _, var_img in variants:
                    reads = ocr_all(var_img)
                    for _, digits in reads:
                        if len(digits) >= 1:
                            grand_votes[digits[0]] += 1

        # Also try reading a TWO-digit zone (positions 10+11 together)
        if target_pos == 10:
            for shift_px in range(-10, 11, 5):
                cx_pair = mx1 - digit_est + shift_px
                pad = digit_est * 1.2
                zx0 = max(0, int(cx_pair - pad))
                zx1 = min(w, int(cx_pair + pad))

                for ch_img in channels.values():
                    zone = ch_img[my0:my1, zx0:zx1] if ch_img.ndim == 2 else ch_img[:, zx0:zx1]
                    if zone.size == 0:
                        continue
                    variants = make_variants(zone, scale=5)
                    for _, var_img in variants:
                        reads = ocr_all(var_img)
                        for _, digits in reads:
                            if len(digits) == 2:
                                # First digit → pos10, second → pos11
                                grand_votes[digits[0]] += 1
                                # (pos11 votes collected separately)

        # Report
        if grand_votes:
            total = sum(grand_votes.values())
            ranked = grand_votes.most_common(8)
            print(f"\n  Total OCR reads: {total}")
            print("  Votes:")
            for d, n in ranked:
                pct = n / total * 100
                bar_graph = "█" * int(pct / 2)
                print(f"    '{d}' : {n:5d} ({pct:5.1f}%)  {bar_graph}")
        else:
            print("  NO READS")

    # ── Step 2: Two-digit pair read (positions 10+11 together) ──
    print(f"\n{'═'*60}")
    print("  TWO-DIGIT PAIR: Positions 10+11 together")
    print(f"{'═'*60}")

    pair_votes = Counter()
    for shift_px in range(-20, 21, 3):
        # The last two hidden digits occupy [mx1 - 2*digit_est, mx1]
        zx0 = max(0, int(mx1 - 2 * digit_est + shift_px - 10))
        zx1 = min(w, int(mx1 + shift_px + 10))

        for ch_img in channels.values():
            zone = ch_img[my0:my1, zx0:zx1] if ch_img.ndim == 2 else ch_img[:, zx0:zx1]
            if zone.size == 0:
                continue
            variants = make_variants(zone, scale=5)
            for _, var_img in variants:
                reads = ocr_all(var_img)
                for _, digits in reads:
                    if len(digits) == 2:
                        pair_votes[digits] += 1
                    elif len(digits) == 1:
                        pair_votes[digits + "?"] += 1

    if pair_votes:
        total = sum(pair_votes.values())
        ranked = pair_votes.most_common(15)
        print(f"\n  Total pair reads: {total}")
        print("  Top pairs:")
        for pair, n in ranked:
            pct = n / total * 100
            print(f"    '{pair}' : {n:5d} ({pct:5.1f}%)")

    # ── Step 3: Read the transition zone (marker→0665) ──
    print(f"\n{'═'*60}")
    print("  TRANSITION ZONE: marker right edge → visible '0665'")
    print(f"{'═'*60}")

    trans_votes = Counter()
    # Read a zone spanning the last ~3 hidden digits + first visible digit
    for shift_px in range(-10, 11, 5):
        zx0 = max(0, int(mx1 - 3 * digit_est + shift_px))
        zx1 = min(w, int(mx1 + 1.5 * digit_est + shift_px))

        for ch_img in channels.values():
            zone = ch_img[my0:my1, zx0:zx1] if ch_img.ndim == 2 else ch_img[:, zx0:zx1]
            if zone.size == 0:
                continue
            variants = make_variants(zone, scale=4)
            for _, var_img in variants:
                reads = ocr_all(var_img)
                for _, digits in reads:
                    if len(digits) >= 3:
                        trans_votes[digits] += 1

    if trans_votes:
        total = sum(trans_votes.values())
        ranked = trans_votes.most_common(20)
        print(f"\n  Total transition reads: {total}")
        print("  Top sequences (expect '??0665' or '???0665'):")
        for seq, n in ranked:
            pct = n / total * 100
            # Highlight if ends with 0665
            tag = " ◄ MATCHES SUFFIX" if seq.endswith("0665") else ""
            tag2 = " ◄ ends 665" if seq.endswith("665") and not tag else ""
            print(f"    '{seq}' : {n:5d} ({pct:5.1f}%){tag}{tag2}")

    # ── Save debug images for positions 10, 11 ──
    print(f"\n{'═'*60}")
    print("  Saving debug images to /tmp/")
    print(f"{'═'*60}")

    for target_pos in [10, 11]:
        # From right edge
        if target_pos == 11:
            cx = mx1 - digit_est * 0.5
        else:
            cx = mx1 - digit_est * 1.5
        pad = digit_est * 0.65
        zx0 = max(0, int(cx - pad))
        zx1 = min(w, int(cx + pad))
        zone = img[my0:my1, zx0:zx1]
        gray_zone = cv2.cvtColor(zone, cv2.COLOR_BGR2GRAY)

        # Save raw zone
        cv2.imwrite(f"/tmp/pos{target_pos}_raw.png", zone)

        # Save best enhancement
        clahe = cv2.createCLAHE(clipLimit=32.0, tileGridSize=(3, 3))
        enhanced = cv2.bitwise_not(clahe.apply(gray_zone))
        big = cv2.resize(
            enhanced,
            (enhanced.shape[1] * 8, enhanced.shape[0] * 8),
            interpolation=cv2.INTER_CUBIC,
        )
        cv2.imwrite(f"/tmp/pos{target_pos}_clahe32_8x.png", big)

        # Morph gradient
        kernel = np.ones((3, 3), np.uint8)
        mg = cv2.morphologyEx(gray_zone, cv2.MORPH_GRADIENT, kernel)
        big_mg = cv2.resize(
            mg,
            (mg.shape[1] * 8, mg.shape[0] * 8),
            interpolation=cv2.INTER_CUBIC,
        )
        cv2.imwrite(f"/tmp/pos{target_pos}_morphgrad_8x.png", big_mg)

        print(
            f"  pos{target_pos}_raw.png, pos{target_pos}_clahe32_8x.png, "
            f"pos{target_pos}_morphgrad_8x.png"
        )

    # Also save the pair zone
    zx0 = max(0, int(mx1 - 2 * digit_est - 10))
    zx1 = min(w, int(mx1 + 10))
    pair_zone = img[my0:my1, zx0:zx1]
    cv2.imwrite("/tmp/pos10_11_pair_raw.png", pair_zone)
    gray_pair = cv2.cvtColor(pair_zone, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=32.0, tileGridSize=(3, 3))
    ep = cv2.bitwise_not(clahe.apply(gray_pair))
    big_pair = cv2.resize(
        ep,
        (ep.shape[1] * 6, ep.shape[0] * 6),
        interpolation=cv2.INTER_CUBIC,
    )
    cv2.imwrite("/tmp/pos10_11_pair_clahe32_6x.png", big_pair)
    print("  pos10_11_pair_raw.png, pos10_11_pair_clahe32_6x.png")


if __name__ == "__main__":
    main()
