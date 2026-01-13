"""Generate synthetic images (clean, blurred, pixelated) for tests."""
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os

OUT_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(OUT_DIR, exist_ok=True)

FONT = None
try:
    FONT = ImageFont.load_default()
except Exception:
    FONT = None


def make_base(text="HELLO WORLD", size=(400, 120)):
    img = Image.new("RGB", size, color=(255, 255, 255))
    d = ImageDraw.Draw(img)
    d.text((10, 40), text, fill=(0, 0, 0), font=FONT)
    path = os.path.join(OUT_DIR, "sample_clean.png")
    img.save(path)
    return path


def pixelate(input_path, block=8):
    img = Image.open(input_path)
    small = img.resize((img.width // block, img.height // block), resample=Image.BILINEAR)
    up = small.resize(img.size, Image.NEAREST)
    out_path = os.path.join(OUT_DIR, "sample_pixelated.png")
    up.save(out_path)
    return out_path


def blur(input_path, radius=3):
    img = Image.open(input_path).filter(Image.Filter.GaussianBlur(radius))
    out_path = os.path.join(OUT_DIR, "sample_blurred.png")
    img.save(out_path)
    return out_path


if __name__ == "__main__":
    base = make_base()
    pixelate(base)
    # blur(base)  # optional
    print("Generated samples in:", OUT_DIR)
