#!/usr/bin/env python3

"""
CLI tool to interact with the Hugging Face Vision Provider for testing and
reconstruction.
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

# Add project root to sys.path for local development
sys.path.append(str(Path(__file__).parent.parent))

from ocr_service.config import get_settings
from ocr_service.modules.ai_providers import HuggingFaceVisionProvider


async def main():
    """
    Main entrypoint for the Hugging Face Vision CLI tool.
    """
    parser = argparse.ArgumentParser(description="Hugging Face Vision CLI")
    parser.add_argument(
        "image_path",
        help="Path to the image file to process",
        type=str,
    )
    parser.add_argument(
        "--prompt",
        default=(
            "Analyze this document image. Identify any obscured, pixelated, or "
            "layered parts. Reconstruct the underlying text."
        ),
        help="Prompt to send to the model",
        type=str,
    )
    parser.add_argument(
        "--token",
        help="Hugging Face Token (overrides HUGGING_FACE_HUB_TOKEN env var)",
        type=str,
        nargs="?",
    )
    parser.add_argument(
        "--model",
        default="runwayml/stable-diffusion-v1-5",
        help="Model ID to use (default: runwayml/stable-diffusion-v1-5)",
        type=str,
    )
    args = parser.parse_args()

    # Get Hugging Face Token from env var or CLI arg
    settings = get_settings()
    token = args.token or settings.hugging_face_hub_token

    if not token:
        print(
            "Error: HUGGING_FACE_HUB_TOKEN is not set in your environment or .env "
            "file, and --token was not provided."
        )
        sys.exit(1)

    # Check if image file exists
    if not os.path.exists(args.image_path):
        print(f"Error: Image file not found at {args.image_path}")
        sys.exit(1)

    # Initialize Hugging Face Vision Provider
    print(f"Initializing HuggingFaceVisionProvider with model {args.model}...")
    provider = HuggingFaceVisionProvider(token=token, model=args.model)

    # Read image file
    print(f"Reading image: {args.image_path}")
    with open(args.image_path, "rb") as f:
        image_bytes = f.read()

    # Send request to Hugging Face
    print(f"Sending request to Hugging Face (prompt: '{args.prompt[:50]}...')...")
    try:
        # Send image file and prompt to the Hugging Face Vision Provider
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
