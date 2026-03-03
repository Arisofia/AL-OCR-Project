"""Run OCR on the most recently modified image file in the workspace."""

import argparse
import asyncio
import json
import re
from pathlib import Path

from ocr_service.modules.ocr_config import EngineConfig
from ocr_service.modules.ocr_engine import IterativeOCREngine

EXCLUDE_PARTS = {".git", ".venv", "node_modules", ".pytest_cache", ".ruff_cache"}
EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def build_parser() -> argparse.ArgumentParser:
    """Build command-line options for OCR probing runs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image-path",
        default="",
        help="Explicit image path to analyze (overrides auto-latest lookup)",
    )
    parser.add_argument(
        "--doc-type",
        default="generic",
        help="Document type hint (e.g. generic, receipt, bank_statement, bank_card)",
    )
    parser.add_argument(
        "--profile",
        default="hybrid",
        choices=["deterministic", "layout_aware", "hybrid"],
        help="OCR strategy profile",
    )
    parser.add_argument(
        "--iterations",
        default=3,
        type=int,
        help="Max OCR iterations",
    )
    return parser


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
    args = build_parser().parse_args()
    image_path = Path(args.image_path) if args.image_path else latest_image_path()
    image_bytes = image_path.read_bytes()

    engine = IterativeOCREngine(
        config=EngineConfig(
            ocr_strategy_profile=args.profile,
            enable_reconstruction=True,
            max_iterations=args.iterations,
        )
    )

    try:
        result = await engine.process_image(
            image_bytes=image_bytes,
            use_reconstruction=True,
            doc_type=args.doc_type,
        )
    finally:
        await engine.close()

    text = result.get("text") or ""
    pan_like = " ".join(re.findall(r"[0-9?]{4}", re.sub(r"\s+", "", text)))
    compact_digits = re.sub(r"[^0-9?]", "", text)

    payload = {
        "image_path": str(image_path),
        "doc_type": args.doc_type,
        "profile": args.profile,
        "success": result.get("success", False),
        "method": result.get("method"),
        "confidence": result.get("confidence"),
        "document_type": result.get("document_type"),
        "type_confidence": result.get("type_confidence"),
        "pan_like_grouped": pan_like,
        "digit_signal": {
            "digits_or_unknowns": len(compact_digits),
            "unknown_count": compact_digits.count("?"),
        },
        "text_preview": (result.get("text") or "")[:1200],
        "iterations": result.get("iterations", []),
        "has_card_analysis": "card_analysis" in result,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
