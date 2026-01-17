#!/usr/bin/env python3

"""
CLI tool to interact with the Hugging Face Vision Provider for testing and
reconstruction.
"""
import asyncio
import argparse
import sys
import os

# Add project root to path to allow importing ocr_service
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)  # noqa: E402
from ocr_service.config import get_settings  # noqa: E402
from ocr_service.modules.ai_providers import HuggingFaceVisionProvider  # noqa: E402


async def main():
    parser = argparse.ArgumentParser(description="Hugging Face Vision CLI")
    parser.add_argument("image_path", help="Path to the image file to process")
    parser.add_argument(
        "--prompt",
        default=(
            "Analyze this document image. Identify any obscured, pixelated, or "
            "layered parts. Reconstruct the underlying text."
        ),
        help="Prompt to send to the model",
    )
    parser.add_argument(
        "--token",
        help="Hugging Face Token (overrides HUGGING_FACE_HUB_TOKEN env var)",
    )
    parser.add_argument(
        "--model",
        default="runwayml/stable-diffusion-v1-5",
        help="Model ID to use (default: runwayml/stable-diffusion-v1-5)",
    )
    args = parser.parse_args()

    settings = get_settings()
    token = args.token or settings.hugging_face_hub_token

    if not token:
        print(
            "Error: HUGGING_FACE_HUB_TOKEN is not set in your environment or .env "
            "file, and --token was not provided."
        )
        sys.exit(1)

    if not os.path.exists(args.image_path):
        print(f"Error: Image file not found at {args.image_path}")
        sys.exit(1)

    print(f"Initializing HuggingFaceVisionProvider with model {args.model}...")
    provider = HuggingFaceVisionProvider(token=token, model=args.model)

    print(f"Reading image: {args.image_path}")
    with open(args.image_path, "rb") as f:
        image_bytes = f.read()

    print(f"Sending request to Hugging Face (prompt: '{args.prompt[:50]}...')...")
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
        print(f"Exception occurred: {e}")


if __name__ == "__main__":
    asyncio.run(main())
