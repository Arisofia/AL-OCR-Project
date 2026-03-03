#!/usr/bin/env python3
"""Find exact pixel position of the '0665' suffix on the card."""

from __future__ import annotations

from typing import Final

import cv2
import pytesseract

IMAGE_PATH: Final = "/Users/jenineferderas/Desktop/card_image.jpg"
OCR_CONFIGS: Final[tuple[str, ...]] = (
    "--oem 3 --psm 7",
    "--oem 3 --psm 6",
    "--oem 3 --psm 11",
)
OCR_EXCEPTIONS = (pytesseract.TesseractError, RuntimeError, TypeError, ValueError)


def _build_digit_boxes(
    box_data: str, x_start: int, source_height: int
) -> list[dict[str, int | str]]:
    """Parse Tesseract box output and keep only digit boxes."""
    boxes: list[dict[str, int | str]] = []
    for line in box_data.strip().split("\n"):
        parts = line.split()
        if len(parts) < 5 or not parts[0].isdigit():
            continue
        boxes.append(
            {
                "ch": parts[0],
                "x1": int(parts[1]) + x_start,
                "x2": int(parts[3]) + x_start,
                "y1": source_height - int(parts[4]),
                "y2": source_height - int(parts[2]),
            }
        )
    return boxes


def main() -> None:
    """Scan multiple right-side crops and report where '0665' is detected."""
    image = cv2.imread(IMAGE_PATH)
    if image is None:
        raise FileNotFoundError(f"Cannot load image: {IMAGE_PATH}")

    image_height, image_width = image.shape[:2]
    y_start, y_end = int(image_height * 0.20), int(image_height * 0.70)
    roi = image[y_start:y_end]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    print(f"Image: {image_width}x{image_height}, ROI y=[{y_start},{y_end}]")

    found = []
    for pct in (0.5, 0.6, 0.65, 0.7):
        x_start = int(image_width * pct)
        crop = gray[:, x_start:]
        for clip in (4, 8, 16, 32):
            clahe = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(4, 4))
            for label, source in (
                ("inv", cv2.bitwise_not(clahe.apply(crop))),
                ("raw", clahe.apply(crop)),
            ):
                for config in OCR_CONFIGS:
                    try:
                        box_data = pytesseract.image_to_boxes(source, config=config)
                    except OCR_EXCEPTIONS:
                        continue

                    boxes = _build_digit_boxes(
                        box_data, x_start=x_start, source_height=source.shape[0]
                    )
                    digit_text = "".join(str(box["ch"]) for box in boxes)
                    if "0665" not in digit_text:
                        continue

                    start_index = digit_text.index("0665")
                    suffix_boxes = boxes[start_index : start_index + 4]
                    pitches = [
                        int(suffix_boxes[i + 1]["x1"]) - int(suffix_boxes[i]["x1"])
                        for i in range(3)
                    ]
                    avg_pitch = sum(pitches) / len(pitches)
                    found.append(
                        {
                            "pct": pct,
                            "clip": clip,
                            "label": label,
                            "config": config,
                            "text": digit_text,
                            "boxes": suffix_boxes,
                            "pitch": avg_pitch,
                        }
                    )

                    print(f"FOUND: pct={pct} clip={clip} {label} {config}")
                    print(f"  text: {digit_text}")
                    for box in suffix_boxes:
                        print(
                            "  "
                            f"'{box['ch']}' x=[{box['x1']}, {box['x2']}] "
                            f"y=[{box['y1']}, {box['y2']}]"
                        )
                    print(f"  pitches: {pitches}, avg={avg_pitch:.1f}")
                    print()

    if found:
        print(f"\n=== SUMMARY: {len(found)} detections ===\n")
        x1s = [int(entry["boxes"][0]["x1"]) for entry in found]
        pitches = [float(entry["pitch"]) for entry in found]
        median_x = sorted(x1s)[len(x1s) // 2]
        median_pitch = sorted(pitches)[len(pitches) // 2]
        print(f"  '0' x1 range: [{min(x1s)}, {max(x1s)}], median={median_x}")
        print(
            "  Pitch range: "
            f"[{min(pitches):.1f}, {max(pitches):.1f}], "
            f"median={median_pitch:.1f}"
        )
        print(f"  Average '0' x1: {sum(x1s) / len(x1s):.1f}")
        print(f"  Average pitch: {sum(pitches) / len(pitches):.1f}")
        return

    print("Could not find '0665' via char boxes")
    print("Trying full-image OCR to see what Tesseract finds...")
    for clip in (8, 32):
        clahe = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(4, 4))
        inverted = cv2.bitwise_not(clahe.apply(gray))
        text = pytesseract.image_to_string(
            inverted,
            config="--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789 ",
        )
        print(f"  clip={clip}: '{text.strip()}'")


if __name__ == "__main__":
    main()
