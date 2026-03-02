"""
Main entry point for the OCR reconstruction library.
Provides a simple CLI to process images and extract text iteratively.
"""

import argparse
import sys

from ocr_reconstruct.modules.pipeline import IterativeOCR


def main():
    """
    Main entrypoint for the OCR reconstruction CLI tool.
    Parses arguments and orchestrates the extraction process.
    """
    parser = argparse.ArgumentParser(description="Iterative OCR Reconstructor")
    parser.add_argument("image_path", help="Path to the image file")
    parser.add_argument(
        "--iterations", type=int, default=3, help="Number of enhancement iterations"
    )
    parser.add_argument(
        "--output-dir", default="./iterations", help="Directory to save debug images"
    )
    parser.add_argument(
        "--save", action="store_true", help="Save intermediate iteration images"
    )

    args = parser.parse_args()

    pipeline = IterativeOCR(
        iterations=args.iterations, save_iterations=args.save, output_dir=args.output_dir
    )

    try:
        text, meta = pipeline.process_file(args.image_path)
        print("\n--- Extracted Text ---")
        print(text)
        print("\n--- Metadata ---")
        print(f"Iterations completed: {len(meta.get('iterations', []))}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
