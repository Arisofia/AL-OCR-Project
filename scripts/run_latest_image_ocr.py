"""Run OCR on the most recently modified image file in the workspace."""

import asyncio
import json
from pathlib import Path

from ocr_service.modules.ocr_config import EngineConfig
from ocr_service.modules.ocr_engine import IterativeOCREngine

EXCLUDE_PARTS = {".git", ".venv", "node_modules", ".pytest_cache", ".ruff_cache"}
EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def latest_image_path() -> Path:
    """Return the newest image path excluding virtualenv and tooling folders."""
    files: list[Path] = []
    for path in Path(".").rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() in EXTS and all(
            part not in EXCLUDE_PARTS for part in path.parts
        ):
            files.append(path)

    if not files:
        raise RuntimeError("NO_IMAGE_FILES_FOUND")

    return max(files, key=lambda image_path: image_path.stat().st_mtime)


async def main() -> None:
    """Execute the OCR engine and print a compact JSON summary."""
    image_path = latest_image_path()
    image_bytes = image_path.read_bytes()

    engine = IterativeOCREngine(
        config=EngineConfig(
            ocr_strategy_profile="hybrid",
            enable_reconstruction=True,
            max_iterations=3,
        )
    )

    try:
        result = await engine.process_image(
            image_bytes=image_bytes,
            use_reconstruction=True,
            doc_type="generic",
        )
    finally:
        await engine.close()

    payload = {
        "image_path": str(image_path),
        "success": result.get("success", False),
        "method": result.get("method"),
        "confidence": result.get("confidence"),
        "document_type": result.get("document_type"),
        "type_confidence": result.get("type_confidence"),
        "text_preview": (result.get("text") or "")[:1200],
        "iterations": result.get("iterations", []),
        "has_card_analysis": "card_analysis" in result,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
