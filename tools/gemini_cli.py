#!/usr/bin/env python3
"""
CLI tool to interact with the Gemini Vision Provider for testing and reconstruction.
"""
import argparse
import asyncio
import os
import sys

try:
    # Try normal imports first (works when package is installed)
    from ocr_service.config import get_settings
    from ocr_service.modules.ai_providers import GeminiVisionProvider
except Exception:  # pragma: no cover - fallback for local dev
    # Fall back to adding project root to sys.path for local development
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from ocr_service.config import get_settings
    from ocr_service.modules.ai_providers import GeminiVisionProvider


async def main():
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
    args = parser.parse_args()

    settings = get_settings()
    if not settings.gemini_api_key:
        print("Error: GEMINI_API_KEY is not set in your environment or .env file.")
        sys.exit(1)

    if not os.path.exists(args.image_path):
        print("Error: Image file not found at", args.image_path)
        sys.exit(1)

    print("Initializing GeminiVisionProvider...")
    provider = GeminiVisionProvider(api_key=settings.gemini_api_key)

    print("Reading image:", args.image_path)
    with open(args.image_path, "rb") as f:
        image_bytes = f.read()

    print(f"Sending request to Gemini (prompt: {args.prompt[:50]}...)")
    try:
        result = await provider.reconstruct(image_bytes, args.prompt)
        if "error" in result:
            print(f"Error from provider: {result['error']}")
        else:
            print("\n--- Reconstructed Text ---")
            print(result.get("text"))
            print("\n--- Metadata ---")
            print(f"Model: {result.get('model')}")
    except Exception as e:
        print("Exception occurred:", e)


if __name__ == "__main__":
    asyncio.run(main())
