# Install as editable package

From the repo root, you can install the reconstruction package in editable mode (recommended for development):

```bash
python -m pip install -e ./ocr_reconstruct
```

This makes `import ocr_reconstruct` available to your Python environment and allows the OCR service to import `ocr_reconstruct.modules.pipeline.process_bytes` directly.

Note: Use a virtualenv to isolate dependencies, and install its requirements:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r ocr_reconstruct/requirements.txt
```