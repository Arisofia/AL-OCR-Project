#!/usr/bin/env python3
"""
Analyze the card image column-by-column to find actual text vs occlusion zones.
Saves intensity profile and diagnostic images.
"""
import cv2
import numpy as np

img = cv2.imread("/Users/jenineferderas/Desktop/card_image.jpg")
h, w = img.shape[:2]
print(f"Image: {w}x{h}")

# Card number Y region (embossed text)
y0, y1 = int(h * 0.25), int(h * 0.55)
number_strip = img[y0:y1]
gray = cv2.cvtColor(number_strip, cv2.COLOR_BGR2GRAY)
print(f"Number strip: y=[{y0},{y1}], height={y1-y0}")

# Column statistics
col_mean = np.mean(gray, axis=0)
col_std = np.std(gray, axis=0)
col_max = np.max(gray, axis=0)
col_min = np.min(gray, axis=0)
col_range = col_max.astype(float) - col_min.astype(float)

# Overall stats
print(f"\nOverall: mean={np.mean(col_mean):.1f}, std of means={np.std(col_mean):.1f}")
print(f"Column range stats: mean={np.mean(col_range):.1f}, std={np.std(col_range):.1f}")

# Find columns with high variability (text regions have more variation)
# vs very uniform columns (solid marker)
var_threshold = np.percentile(col_range, 40)
text_cols = col_range > var_threshold
uniform_cols = col_range <= var_threshold

# Find contiguous text and uniform regions
regions = []
current_type = "text" if text_cols[0] else "uniform"
start = 0
for i in range(1, len(text_cols)):
    new_type = "text" if text_cols[i] else "uniform"
    if new_type != current_type:
        regions.append((current_type, start, i - 1))
        current_type = new_type
        start = i
regions.append((current_type, start, len(text_cols) - 1))

# Filter small regions (noise)
sig_regions = [(t, s, e) for t, s, e in regions if (e - s) > 15]
print(f"\nSignificant regions (width>15px):")
for rtype, rs, re_ in sig_regions:
    w_ = re_ - rs
    avg_bright = np.mean(col_mean[rs:re_+1])
    print(f"  x=[{rs:4d}, {re_:4d}] width={w_:4d}  type={rtype:7s}  avg_bright={avg_bright:.1f}")

# Look for the darkest contiguous region (likely the marker/occlusion)
print(f"\nLooking for darkest uniform regions (potential occlusion marker):")
uniform_regs = [(t, s, e) for t, s, e in sig_regions if t == "uniform" and (e - s) > 30]
for _, s, e in sorted(uniform_regs, key=lambda x: np.mean(col_mean[x[1]:x[2]+1])):
    avg = np.mean(col_mean[s:e+1])
    w_ = e - s
    print(f"  x=[{s:4d}, {e:4d}] width={w_:4d}  avg_brightness={avg:.1f}")

# Enhanced approach: look at CLAHE enhanced image
clahe = cv2.createCLAHE(clipLimit=16.0, tileGridSize=(4, 4))
enh = clahe.apply(gray)
col_mean_enh = np.mean(enh, axis=0)
col_std_enh = np.std(enh, axis=0)

print(f"\nAfter CLAHE enhancement:")
print(f"  Column std stats: mean={np.mean(col_std_enh):.1f}, max={np.max(col_std_enh):.1f}")

# Find columns where CLAHE std is very low (dead zone = heavy occlusion)
low_var_mask = col_std_enh < np.percentile(col_std_enh, 20)
dead_regions = []
in_dead = False
ds = 0
for i in range(len(low_var_mask)):
    if low_var_mask[i] and not in_dead:
        ds = i
        in_dead = True
    elif not low_var_mask[i] and in_dead:
        if i - ds > 20:
            dead_regions.append((ds, i - 1))
        in_dead = False
if in_dead and len(low_var_mask) - ds > 20:
    dead_regions.append((ds, len(low_var_mask) - 1))

print(f"\nLow-variance (dead) regions after CLAHE:")
for ds, de in dead_regions:
    print(f"  x=[{ds:4d}, {de:4d}] width={de-ds:4d}")

# Save diagnostic images
# 1. The number strip raw
cv2.imwrite("/tmp/card_number_strip.png", number_strip)

# 2. CLAHE enhanced
cv2.imwrite("/tmp/card_number_clahe.png", enh)

# 3. Column intensity plot as image
plot_h = 200
plot = np.ones((plot_h, w, 3), dtype=np.uint8) * 255
for x in range(w):
    y_mean = int((1 - col_mean[x] / 255.0) * (plot_h - 1))
    cv2.line(plot, (x, plot_h - 1), (x, y_mean), (0, 0, 0), 1)
    y_range = int((col_range[x] / 255.0) * (plot_h - 1))
    cv2.circle(plot, (x, plot_h - 1 - y_range), 1, (0, 0, 255), -1)
    y_std = int((col_std_enh[x] / 80.0) * (plot_h - 1))
    cv2.circle(plot, (x, plot_h - 1 - y_std), 1, (0, 255, 0), -1)
cv2.imwrite("/tmp/card_intensity_profile.png", plot)

# 4. Overlay column variance on the strip for visual alignment
overlay = cv2.cvtColor(enh, cv2.COLOR_GRAY2BGR)
for x in range(w):
    if low_var_mask[x]:  # dead zone - mark red
        overlay[:, x, 2] = np.minimum(overlay[:, x, 2].astype(int) + 80, 255).astype(np.uint8)
cv2.imwrite("/tmp/card_dead_zones.png", overlay)

print("\nDiagnostic images:")
print("  /tmp/card_number_strip.png")
print("  /tmp/card_number_clahe.png")
print("  /tmp/card_intensity_profile.png  (black=mean, red=range, green=CLAHE std)")
print("  /tmp/card_dead_zones.png  (red tint = low variance / dead zone)")
