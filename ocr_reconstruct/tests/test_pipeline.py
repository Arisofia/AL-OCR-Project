import os

from ocr_reconstruct.modules.pipeline import IterativeOCR


def test_pipeline_on_pixelated_sample(tmp_path):
    # Ensure sample exists
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    img_path = os.path.join(data_dir, "sample_pixelated.png")
    assert os.path.exists(img_path), (
        "Generate sample images by running tests/generate_samples.py"
    )

    worker = IterativeOCR(iterations=2, save_iterations=True, output_dir=str(tmp_path))
    text, _meta = worker.process_file(img_path)
    assert isinstance(text, str)
    # We expect at least some characters recovered for synthetic sample
    assert len(text) >= 1
