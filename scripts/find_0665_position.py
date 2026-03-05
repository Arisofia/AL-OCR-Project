"""Find exact pixel position of the '0665' suffix on the card."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import cv2
import pytesseract

IMAGE_PATH: Final = "/Users/jenineferderas/Desktop/card_image.jpg"
OCR_CONFIGS: Final[tuple[str, ...]] = (
    "--oem 3 --psm 7",
    "--oem 3 --psm 6",
    "--oem 3 --psm 11",
)
PCT_CANDIDATES: Final[tuple[float, ...]] = (0.5, 0.6, 0.65, 0.7)
CLIP_CANDIDATES: Final[tuple[int, ...]] = (4, 8, 16, 32)
OCR_EXCEPTIONS = (pytesseract.TesseractError, RuntimeError, TypeError, ValueError)


@dataclass(frozen=True)
class DigitBox:
    """OCR digit bounding box in ROI coordinates."""

    ch: str
    x1: int
    x2: int
    y1: int
    y2: int


@dataclass(frozen=True)
class Detection:
    """One successful detection of the 0665 suffix."""

    pct: float
    clip: int
    label: str
    config: str
    text: str
    boxes: tuple[DigitBox, ...]
    pitch: float


@dataclass(frozen=True)
class SourceVariant:
    """One preprocessed crop plus metadata used for OCR."""

    pct: float
    clip: int
    label: str
    source: cv2.typing.MatLike
    x_start: int


def _build_digit_boxes(
    box_data: str, x_start: int, source_height: int
) -> list[DigitBox]:
    """Parse Tesseract box output and keep only digit boxes."""
    boxes: list[DigitBox] = []
    for line in box_data.strip().split("\n"):
        parts = line.split()
        if len(parts) < 5 or not parts[0].isdigit():
            continue
        boxes.append(
            DigitBox(
                ch=parts[0],
                x1=int(parts[1]) + x_start,
                x2=int(parts[3]) + x_start,
                y1=source_height - int(parts[4]),
                y2=source_height - int(parts[2]),
            )
        )
    return boxes


def _load_gray_roi() -> tuple[int, int, int, int, cv2.typing.MatLike]:
    """Load image and return dimensions plus grayscale ROI."""
    image = cv2.imread(IMAGE_PATH)
    if image is None:
        raise FileNotFoundError(f"Cannot load image: {IMAGE_PATH}")

    image_height, image_width = image.shape[:2]
    y_start, y_end = int(image_height * 0.20), int(image_height * 0.70)
    roi = image[y_start:y_end]
    return image_height, image_width, y_start, y_end, cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)


def _iter_sources(gray: cv2.typing.MatLike, image_width: int):
    """Yield preprocessed right-side crops with metadata."""
    for pct in PCT_CANDIDATES:
        x_start = int(image_width * pct)
        crop = gray[:, x_start:]
        for clip in CLIP_CANDIDATES:
            clahe = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(4, 4))
            raw = clahe.apply(crop)
            yield SourceVariant(
                pct=pct,
                clip=clip,
                label="inv",
                source=cv2.bitwise_not(raw),
                x_start=x_start,
            )
            yield SourceVariant(
                pct=pct, clip=clip, label="raw", source=raw, x_start=x_start
            )


def _build_detection(
    variant: SourceVariant,
    config: str,
) -> Detection | None:
    """Try to detect 0665 in one OCR source/config combination."""
    try:
        box_data = pytesseract.image_to_boxes(variant.source, config=config)
    except OCR_EXCEPTIONS:
        return None

    boxes = _build_digit_boxes(
        box_data=box_data,
        x_start=variant.x_start,
        source_height=variant.source.shape[0],
    )
    digit_text = "".join(box.ch for box in boxes)
    if "0665" not in digit_text:
        return None

    start_index = digit_text.index("0665")
    suffix_boxes = tuple(boxes[start_index : start_index + 4])
    if len(suffix_boxes) < 4:
        return None

    pitches = [
        suffix_boxes[index + 1].x1 - suffix_boxes[index].x1 for index in range(3)
    ]
    avg_pitch = sum(pitches) / len(pitches)
    return Detection(
        pct=variant.pct,
        clip=variant.clip,
        label=variant.label,
        config=config,
        text=digit_text,
        boxes=suffix_boxes,
        pitch=avg_pitch,
    )


def _print_detection(detection: Detection) -> None:
    """Print detailed detection info."""
    pitches = [
        detection.boxes[index + 1].x1 - detection.boxes[index].x1
        for index in range(3)
    ]
    print(
        f"FOUND: pct={detection.pct} clip={detection.clip} "
        f"{detection.label} {detection.config}"
    )
    print(f"  text: {detection.text}")
    for box in detection.boxes:
        print(f"  '{box.ch}' x=[{box.x1}, {box.x2}] y=[{box.y1}, {box.y2}]")
    print(f"  pitches: {pitches}, avg={detection.pitch:.1f}")
    print()


def _print_summary(found: list[Detection]) -> None:
    """Print aggregated detection statistics."""
    print(f"\n=== SUMMARY: {len(found)} detections ===\n")
    x1s = [entry.boxes[0].x1 for entry in found]
    pitches = [entry.pitch for entry in found]
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


def _print_fallback_ocr(gray: cv2.typing.MatLike) -> None:
    """Run and print fallback full-line OCR when 0665 wasn't found by boxes."""
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


def main() -> None:
    """Scan multiple right-side crops and report where '0665' is detected."""
    image_height, image_width, y_start, y_end, gray = _load_gray_roi()
    print(f"Image: {image_width}x{image_height}, ROI y=[{y_start},{y_end}]")

    found: list[Detection] = []
    for variant in _iter_sources(gray, image_width):
        for config in OCR_CONFIGS:
            detection = _build_detection(variant=variant, config=config)
            if detection is None:
                continue
            found.append(detection)
            _print_detection(detection)

    if found:
        _print_summary(found)
        return

    _print_fallback_ocr(gray)


if __name__ == "__main__":
    main()
