#!/usr/bin/env python3
"""
CLI tool to interact with the Gemini Vision Provider for testing and reconstruction.
"""

import argparse
import asyncio
import contextlib
import importlib
import sys
from pathlib import Path

# Add project root to sys.path for local development
with contextlib.suppress(Exception):
    sys.path.append(str(Path(__file__).parent.parent))

get_settings = importlib.import_module("ocr_service.config").get_settings
_providers_mod = importlib.import_module("ocr_service.modules.ai_providers")
AIProviderError = _providers_mod.AIProviderError
GeminiVisionProvider = _providers_mod.GeminiVisionProvider


async def main():
    """Run Gemini Vision OCR reconstruction from the command line."""
    parser = argparse.ArgumentParser(description="Gemini Vision CLI")
    parser.add_argument("image_path", help="Path to the image file to process")
    parser.add_argument(
        "--prompt",
        default=(
            "Analyze this document image. Identify any obscured, pixelated, "
            "or layered parts. Reconstruct the underlying text."
        ),
        help="Prompt to send to the model",
    )
    parser.add_argument(
        "--api-key",
        help="Gemini API Key (overrides GEMINI_API_KEY env var)",
    )
    args = parser.parse_args()

    settings = get_settings()
    api_key = args.api_key or settings.gemini_api_key

    if not api_key:
        print(
            "Error: GEMINI_API_KEY is not set in your environment or .env file, "
            "and --api-key was not provided."
        )
        sys.exit(1)

    image_path = Path(args.image_path)
    if not image_path.exists():
        print("Error: Image file not found at", args.image_path)
        sys.exit(1)

    print("Initializing GeminiVisionProvider...")
    provider = GeminiVisionProvider(api_key=api_key)

    print("Reading image:", args.image_path)
    image_bytes = await asyncio.to_thread(image_path.read_bytes)

    print(f"Sending request to Gemini (prompt: {args.prompt[:50]}...)")
    try:
        result = await provider.reconstruct(image_bytes, args.prompt)
        if "error" in result:
            print(f"Error from provider: {result['error']}")
            if result.get("detail"):
                print("Detail:", result.get("detail"))
            if result.get("body") is not None:
                print("Body:", result.get("body"))
        else:
            print("\n--- Reconstructed Text ---")
            print(result.get("text"))
            print("\n--- Metadata ---")
            print(f"Model: {result.get('model')}")
    except (AIProviderError, OSError, ValueError) as e:
        print("Exception occurred:", e)


if __name__ == "__main__":
    asyncio.run(main())
