# Iterative OCR with Pixel Reconstruction

A concise, research-oriented Python project that implements an iterative OCR pipeline combined with pixel reconstruction and image enhancement techniques to recover partially obscured text.

IMPORTANT: This project is for research, education, and lawful uses only. Do not use it to attempt to reverse intentional redactions or to breach privacy.


Features

- Iterative pipeline using OpenCV for preprocessing (grayscale, sharpening, denoising, thresholding) and Tesseract for OCR
- Basic pixel reconstruction heuristics (upsampling + smoothing) and inpainting for partial occlusions
- CLI for single-image and batch runs
- Synthetic data generator for tests (pixelation/blur overlays)
- Pytest unit tests demonstrating expected behavior on synthetic inputs


Quickstart

1. Create a Python 3.8+ virtualenv and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Ensure Tesseract is installed and available on PATH. On macOS: `brew install tesseract`.

3. Run CLI:

```bash
python main.py --image tests/data/sample_clean.png --iterations 3 --output out.txt
```

License & Ethics
- Licensed for research and evaluation; see `LICENSE`.
- Include an ethical disclaimer: recovery from strong occlusions is often impossible; always obtain consent before processing sensitive images.
