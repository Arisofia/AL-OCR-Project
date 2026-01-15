import os

from ocr_reconstruct.modules.pipeline import process_bytes


def test_import_and_process_sample():
    # Generate samples if not present
    gen = os.path.join(os.path.dirname(__file__), "generate_samples.py")
    if os.path.exists(gen):
        os.system(f"python {gen}")

    sample = os.path.join(
        os.path.dirname(__file__),
        "data",
        "sample_pixelated.png",
    )
    assert os.path.exists(sample), (
        "Sample image missing; ensure generate_samples.py was run"
    )

    with open(sample, "rb") as fh:
        img_bytes = fh.read()
        text, _out_bytes, meta = process_bytes(img_bytes, iterations=1)

    assert isinstance(text, str)
    # out_bytes may be None if encoding failed, but meta should be present
    assert "iterations" in meta
