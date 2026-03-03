#!/usr/bin/env python3
"""Find exact pixel position of '0665' on the card."""
import cv2
import numpy as np
import pytesseract

img = cv2.imread("/Users/jenineferderas/Desktop/card_image.jpg")
h, w = img.shape[:2]
y0, y1 = int(h * 0.20), int(h * 0.70)
roi = img[y0:y1]
gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

print(f"Image: {w}x{h}, ROI y=[{y0},{y1}]")

found = []
for pct in [0.5, 0.6, 0.65, 0.7]:
    x_start = int(w * pct)
    crop = gray[:, x_start:]
    for clip in [4, 8, 16, 32]:
        c = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(4, 4))
        for label, src in [("inv", cv2.bitwise_not(c.apply(crop))), ("raw", c.apply(crop))]:
            for cfg in ["--oem 3 --psm 7", "--oem 3 --psm 6", "--oem 3 --psm 11"]:
                try:
                    data = pytesseract.image_to_boxes(src, config=cfg)
                    boxes = []
                    for line in data.strip().split("\n"):
                        parts = line.split()
                        if len(parts) >= 5 and parts[0].isdigit():
                            boxes.append({
                                "ch": parts[0],
                                "x1": int(parts[1]) + x_start,
                                "x2": int(parts[3]) + x_start,
                                "y1": src.shape[0] - int(parts[4]),
                                "y2": src.shape[0] - int(parts[2]),
                            })
                    dtxt = "".join(b["ch"] for b in boxes)
                    if "0665" in dtxt:
                        idx = dtxt.index("0665")
                        suffix_boxes = boxes[idx : idx + 4]
                        pitches = [suffix_boxes[i + 1]["x1"] - suffix_boxes[i]["x1"] for i in range(3)]
                        avg_pitch = sum(pitches) / len(pitches)
                        entry = {
                            "pct": pct,
                            "clip": clip,
                            "label": label,
                            "cfg": cfg,
                            "text": dtxt,
                            "boxes": suffix_boxes,
                            "pitch": avg_pitch,
                        }
                        found.append(entry)
                        print(
                            f"FOUND: pct={pct} clip={clip} {label} {cfg}"
                        )
                        print(f"  text: {dtxt}")
                        for b in suffix_boxes:
                            print(f"  '{b['ch']}' x=[{b['x1']}, {b['x2']}] y=[{b['y1']}, {b['y2']}]")
                        print(f"  pitches: {pitches}, avg={avg_pitch:.1f}")
                        print()
                except Exception:
                    pass

if found:
    print(f"\n=== SUMMARY: {len(found)} detections ===\n")
    x1s = [f["boxes"][0]["x1"] for f in found]
    pitches = [f["pitch"] for f in found]
    print(f"  '0' x1 range: [{min(x1s)}, {max(x1s)}], median={sorted(x1s)[len(x1s)//2]}")
    print(f"  Pitch range: [{min(pitches):.1f}, {max(pitches):.1f}], median={sorted(pitches)[len(pitches)//2]:.1f}")
    print(f"  Average '0' x1: {sum(x1s)/len(x1s):.1f}")
    print(f"  Average pitch: {sum(pitches)/len(pitches):.1f}")
else:
    print("Could not find '0665' via char boxes")
    print("Trying full-image OCR to see what Tesseract finds...")
    for clip in [8, 32]:
        c = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(4, 4))
        inv = cv2.bitwise_not(c.apply(gray))
        txt = pytesseract.image_to_string(inv, config="--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789 ")
        print(f"  clip={clip}: '{txt.strip()}'")
