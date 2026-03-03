#!/usr/bin/env python3
"""
Rescan all 6 hidden positions using CORRECT coordinates from column analysis.

From card_column_analysis.py:
  - Visible suffix "0665": '0' at ~1132, pitch ~75px
  - Group gap (group3→group4): x≈1085-1131 (~47px)
  - Position 11 text visible at x=[1032,1084], center≈1058
  - Heavy marker: x=[815,1031]
  - Position 10 center ≈ 1058-75 = 983  (under marker)
  - Position 9 center ≈ 908  (under marker)
  - Position 8 center ≈ 833  (under marker)
  - Group gap (group2→group3): estimated at ~45px around x=790-833
  - Position 7 center ≈ 708  (barely visible, brightness 23)
  - Position 6 center ≈ 647  (partially visible, brightness 30)
"""
import re
from collections import Counter
import cv2
import numpy as np
import pytesseract

IMAGE_PATH = "/Users/jenineferderas/Desktop/card_image.jpg"
TESS_NUMERIC_ONLY = "-c tessedit_char_whitelist=0123456789"
CFG_SINGLE_DIGIT = f"--oem 3 --psm 13 {TESS_NUMERIC_ONLY}"
CFG_GROUP = f"--oem 3 --psm 7 {TESS_NUMERIC_ONLY}"

# Digit centers derived from column analysis
CENTERS = {
    6: 647,
    7: 708,
    8: 833,
    9: 908,
    10: 983,
    11: 1058,
}
PITCH = 75  # pixel width per digit (from visible suffix analysis)


def enhance(img_gray, scale=5):
    """Generate multiple enhanced variants of an image zone."""
    h0, w0 = img_gray.shape

    def upscaled_res(image):
        return cv2.resize(image, (w0 * scale, h0 * scale), interpolation=cv2.INTER_CUBIC)

    out = []
    for clip in [4, 8, 16, 32, 64]:
        clahe_obj = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(3, 3))
        applied = clahe_obj.apply(img_gray)
        out.extend([upscaled_res(cv2.bitwise_not(applied)), upscaled_res(applied)])

    k = np.ones((3, 3), np.uint8)
    out.extend(
        [
            upscaled_res(cv2.morphologyEx(img_gray, cv2.MORPH_GRADIENT, k)),
            upscaled_res(cv2.bitwise_not(cv2.equalizeHist(img_gray))),
        ]
    )

    for gamma in [0.3, 0.5]:
        lut = np.array(
            [((i / 255.0) ** gamma) * 255 for i in range(256)]
        ).astype(np.uint8)
        out.append(upscaled_res(cv2.bitwise_not(cv2.LUT(img_gray, lut))))

    # Top-hat
    for ks in [7, 11]:
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
        out.append(upscaled_res(cv2.morphologyEx(img_gray, cv2.MORPH_TOPHAT, kern)))
    return out


def ocr_single(img_var):  # NOSONAR
    """Run multiple single-digit OCR attempts on an image variant."""
    results = []
    ocr_configs = [
        f"--oem 3 --psm 10 {TESS_NUMERIC_ONLY}",
        CFG_SINGLE_DIGIT,
        f"--oem 3 --psm 8 {TESS_NUMERIC_ONLY}",
    ]
    for config in ocr_configs:
        try:
            if text := pytesseract.image_to_string(img_var, config=config).strip():
                if digits := re.sub(r"\D", "", text):
                    results.append(digits[0])
        except Exception:  # pylint: disable=broad-except
            pass
        for th_val in [100, 130, 160]:
            _, bw_img = cv2.threshold(img_var, th_val, 255, cv2.THRESH_BINARY)
            try:
                if t_bw := pytesseract.image_to_string(bw_img, config=config).strip():
                    if d_bw := re.sub(r"\D", "", t_bw):
                        results.append(d_bw[0])
            except Exception:  # pylint: disable=broad-except
                pass
    return results


def scan_position(pos, img_w, channels):
    """Perform multi-channel, multi-offset scan for a single digit position."""
    cx_val = CENTERS[pos]
    votes = Counter()
    for dx in [-10, -5, 0, 5, 10]:
        actual_cx = cx_val + dx
        half_w = PITCH * 0.5
        zx0 = max(0, int(actual_cx - half_w))
        zx1 = min(img_w, int(actual_cx + half_w))

        for ch_img in channels.values():
            zone = ch_img[:, zx0:zx1]
            if zone.size == 0:
                continue
            for variant in enhance(zone):
                for digit in ocr_single(variant):
                    votes[digit] += 1
    return votes


def main():  # NOSONAR
    """Main execution entry point."""
    orig_img = cv2.imread(IMAGE_PATH)
    if orig_img is None:
        raise FileNotFoundError(f"Could not read image at {IMAGE_PATH}")

    img_h, img_w = orig_img.shape[:2]
    y0_roi, y1_roi = int(img_h * 0.25), int(img_h * 0.55)
    roi_img = orig_img[y0_roi:y1_roi]
    gray_roi = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
    print(f"Image: {img_w}x{img_h}, ROI y=[{y0_roi},{y1_roi}]")

    img_channels = {
        "gray": gray_roi,
        "blue": roi_img[:, :, 0],
        "green": roi_img[:, :, 1],
        "red": roi_img[:, :, 2],
    }

    print("\n=== Per-position scan (corrected coordinates) ===\n")
    for pos_idx in range(6, 12):
        cx_target = CENTERS[pos_idx]
        pos_votes = scan_position(pos_idx, img_w, img_channels)
        total_votes = sum(pos_votes.values())

        if total_votes:
            ranked = pos_votes.most_common(6)
            top_res = ", ".join(f"'{d}'={n/total_votes*100:.0f}%" for d, n in ranked[:5])
            note = ""
            if 815 <= cx_target <= 1031:
                note = " [under marker]"
            elif 700 < cx_target < 815:
                note = " [barely visible]"
            elif 1032 <= cx_target <= 1084:
                note = " [edge visible]"
            print(f"  POS {pos_idx} (cx={cx_target}){note}: {top_res}  (n={total_votes})")
        else:
            print(f"  POS {pos_idx} (cx={cx_target}): no reads")

    # ── Pair reads for positions 10+11 ──
    print("\n=== PAIR: Positions 10+11 together ===")
    pair_v = Counter()
    for dx in [-15, -5, 5, 15]:
        p_cx = (CENTERS[10] + CENTERS[11]) / 2 + dx
        p_hw = PITCH * 1.1
        zx0, zx1 = max(0, int(p_cx - p_hw)), min(img_w, int(p_cx + p_hw))
        for ch_img in img_channels.values():
            p_zone = ch_img[:, zx0:zx1]
            if p_zone.size == 0:
                continue
            for var in enhance(p_zone):
                for p_cfg in [CFG_GROUP, CFG_SINGLE_DIGIT]:
                    try:
                        if txt := pytesseract.image_to_string(var, config=p_cfg).strip():
                            if p_digits := re.sub(r"\D", "", txt):
                                if len(p_digits) == 2:
                                    pair_v[p_digits] += 1
                    except Exception:  # pylint: disable=broad-except
                        pass
                    for th_v in [120, 160]:
                        _, p_bw = cv2.threshold(var, th_v, 255, cv2.THRESH_BINARY)
                        try:
                            if t_p_bw := pytesseract.image_to_string(p_bw, config=p_cfg).strip():
                                if d_p_bw := re.sub(r"\D", "", t_p_bw):
                                    if len(d_p_bw) == 2:
                                        pair_v[d_p_bw] += 1
                        except Exception:  # pylint: disable=broad-except
                            pass

    total_pair = sum(pair_v.values())
    if total_pair:
        for pair_seq, n_votes in pair_v.most_common(10):
            print(f"  '{pair_seq}': {n_votes:4d} ({n_votes/total_pair*100:5.1f}%)")

    # ── Save Debug Images ──
    print("\n=== Debug images ===")
    dbg_hw = int(PITCH * 0.6)
    c11, c10 = CENTERS[11], CENTERS[10]
    z11 = gray_roi[:, max(0, c11 - dbg_hw) : min(img_w, c11 + dbg_hw)]
    z10 = gray_roi[:, max(0, c10 - dbg_hw) : min(img_w, c10 + dbg_hw)]
    cv2.imwrite("/tmp/pos11_correct_raw.png", z11)
    cv2.imwrite("/tmp/pos10_correct_raw.png", z10)
    clahe_dbg = cv2.createCLAHE(clipLimit=32.0, tileGridSize=(3, 3))
    for name, zone in [("pos11", z11), ("pos10", z10)]:
        e_img = cv2.bitwise_not(clahe_dbg.apply(zone))
        b_img = cv2.resize(
            e_img, (e_img.shape[1] * 8, e_img.shape[0] * 8), interpolation=cv2.INTER_CUBIC
        )
        cv2.imwrite(f"/tmp/{name}_correct_enh.png", b_img)
    print("  Debug images saved to /tmp/")


if __name__ == "__main__":
    main()
