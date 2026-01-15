"""Generate synthetic images (clean, blurred, pixelated) for tests."""

import os

from PIL import Image, ImageDraw, ImageFilter, ImageFont

try:
    RESAMPLING = Image.Resampling
except AttributeError:
    # Older Pillow versions
    RESAMPLING = Image  # type: ignore

OUT_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(OUT_DIR, exist_ok=True)

try:
    DEFAULT_FONT = ImageFont.load_default()
except Exception:  # pylint: disable=broad-exception-caught
    DEFAULT_FONT = None  # type: ignore


def make_base(text="HELLO WORLD", size=(400, 120)):
    """Creates a base image with text."""
    img = Image.new("RGB", size, color=(255, 255, 255))
    d = ImageDraw.Draw(img)
    d.text((10, 40), text, fill=(0, 0, 0), font=DEFAULT_FONT)
    path = os.path.join(OUT_DIR, "sample_clean.png")
    img.save(path)
    return path


def pixelate(input_path, block=8):
    """Applies pixelation to an image."""
    img = Image.open(input_path)
    small = img.resize(
        (img.width // block, img.height // block),
        resample=RESAMPLING.BILINEAR,
    )
    up = small.resize(img.size, RESAMPLING.NEAREST)
    out_path = os.path.join(OUT_DIR, "sample_pixelated.png")
    up.save(out_path)
    return out_path


def blur(input_path, radius=3):
    """Applies Gaussian blur to an image."""
    img = Image.open(input_path).filter(ImageFilter.GaussianBlur(radius))
    out_path = os.path.join(OUT_DIR, "sample_blurred.png")
    img.save(out_path)
    return out_path


if __name__ == "__main__":
    base = make_base()
    pixelate(base)
    # blur(base)  # optional
    print("Generated samples in:", OUT_DIR)
