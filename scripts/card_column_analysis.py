"""
Analyze the card image column-by-column to find actual text vs occlusion zones.
Saves intensity profile and diagnostic images.
"""
import cv2
import numpy as np

IMAGE_PATH = "/Users/jenineferderas/Desktop/card_image.jpg"


def analyze_card_columns():
    """Perform column-wise intensity analysis on the card image strip."""
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        raise FileNotFoundError(f"Could not read image at {IMAGE_PATH}")

    img_h, img_w = img.shape[:2]
    print(f"Image: {img_w}x{img_h}")

    y0_roi, y1_roi = int(img_h * 0.25), int(img_h * 0.55)
    number_strip = img[y0_roi:y1_roi]
    gray = cv2.cvtColor(number_strip, cv2.COLOR_BGR2GRAY)
    print(f"Number strip: y=[{y0_roi},{y1_roi}], height={y1_roi-y0_roi}")

    col_mean = np.mean(gray, axis=0)
    col_max = np.max(gray, axis=0)
    col_min = np.min(gray, axis=0)
    col_range = col_max.astype(float) - col_min.astype(float)

    print(f"\nOverall: mean={np.mean(col_mean):.1f}, std of means={np.std(col_mean):.1f}")
    print(f"Column range stats: mean={np.mean(col_range):.1f}, std={np.std(col_range):.1f}")

    var_threshold = np.percentile(col_range, 40)
    text_cols = col_range > var_threshold

    regions = []
    current_region_type = "text" if text_cols[0] else "uniform"
    region_start = 0
    for i in range(1, len(text_cols)):
        new_region_type = "text" if text_cols[i] else "uniform"
        if new_region_type != current_region_type:
            regions.append((current_region_type, region_start, i - 1))
            current_region_type = new_region_type
            region_start = i
    regions.append((current_region_type, region_start, len(text_cols) - 1))

    sig_regions = [(t, s, e) for t, s, e in regions if (e - s) > 15]
    print("\nSignificant regions (width>15px):")
    for rtype, rs, re_ in sig_regions:
        w_len = re_ - rs
        avg_bright = np.mean(col_mean[rs : re_ + 1])
        print(
            f"  x=[{rs:4d}, {re_:4d}] width={w_len:4d}  "
            f"type={rtype:7s}  avg_bright={avg_bright:.1f}"
        )

    print("\nLooking for darkest uniform regions (potential occlusion marker):")
    uniform_regs = [(t, s, e) for t, s, e in sig_regions if t == "uniform" and (e - s) > 30]
    for _, s, e in sorted(uniform_regs, key=lambda x: np.mean(col_mean[x[1] : x[2] + 1])):
        avg = np.mean(col_mean[s : e + 1])
        w_len = e - s
        print(f"  x=[{s:4d}, {e:4d}] width={w_len:4d}  avg_brightness={avg:.1f}")

    clahe = cv2.createCLAHE(clipLimit=16.0, tileGridSize=(4, 4))
    enh = clahe.apply(gray)
    col_std_enh = np.std(enh, axis=0)

    print("\nAfter CLAHE enhancement:")
    print(f"  Column std stats: mean={np.mean(col_std_enh):.1f}, max={np.max(col_std_enh):.1f}")

    low_var_mask = col_std_enh < np.percentile(col_std_enh, 20)
    dead_regions = []
    is_in_dead_zone = False
    dead_start = 0
    for i, is_low_var in enumerate(low_var_mask):
        if is_low_var and not is_in_dead_zone:
            dead_start = i
            is_in_dead_zone = True
        elif not is_low_var and is_in_dead_zone:
            if i - dead_start > 20:
                dead_regions.append((dead_start, i - 1))
            is_in_dead_zone = False
    if is_in_dead_zone and len(low_var_mask) - dead_start > 20:
        dead_regions.append((dead_start, len(low_var_mask) - 1))

    print("\nLow-variance (dead) regions after CLAHE:")
    for d_start, d_end in dead_regions:
        print(f"  x=[{d_start:4d}, {d_end:4d}] width={d_end-d_start:4d}")

    cv2.imwrite("/tmp/card_number_strip.png", number_strip)
    cv2.imwrite("/tmp/card_number_clahe.png", enh)

    plot_height = 200
    plot = np.ones((plot_height, img_w, 3), dtype=np.uint8) * 255
    for x in range(img_w):
        y_mean = int((1 - col_mean[x] / 255.0) * (plot_height - 1))
        cv2.line(plot, (x, plot_height - 1), (x, y_mean), (0, 0, 0), 1)
        y_range = int((col_range[x] / 255.0) * (plot_height - 1))
        cv2.circle(plot, (x, plot_height - 1 - y_range), 1, (0, 0, 255), -1)
        y_std = int((col_std_enh[x] / 80.0) * (plot_height - 1))
        cv2.circle(plot, (x, plot_height - 1 - y_std), 1, (0, 255, 0), -1)
    cv2.imwrite("/tmp/card_intensity_profile.png", plot)

    overlay = cv2.cvtColor(enh, cv2.COLOR_GRAY2BGR)
    for x in range(img_w):
        if low_var_mask[x]:
            overlay[:, x, 2] = np.minimum(overlay[:, x, 2].astype(int) + 80, 255).astype(np.uint8)
    cv2.imwrite("/tmp/card_dead_zones.png", overlay)

    print("\nDiagnostic images:")
    print("  /tmp/card_number_strip.png")
    print("  /tmp/card_number_clahe.png")
    print("  /tmp/card_intensity_profile.png  (black=mean, red=range, green=CLAHE std)")
    print("  /tmp/card_dead_zones.png  (red tint = low variance / dead zone)")


if __name__ == "__main__":
    analyze_card_columns()
