"""CLI entrypoint for the iterative OCR + reconstruction project."""
import argparse
from modules.pipeline import IterativeOCR


def parse_args():
    p = argparse.ArgumentParser(
        description="Iterative OCR with pixel reconstruction",
    )
    p.add_argument(
        "--image",
        help="Path to the image file",
        required=True,
    )
    p.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="Number of iterations",
    )
    p.add_argument(
        "--output",
        default="reconstructed_text.txt",
        help="Output text file",
    )
    p.add_argument(
        "--save-iterations",
        action="store_true",
        help="Save images of each iteration",
    )
    return p.parse_args()


def main():
    args = parse_args()
    worker = IterativeOCR(
        iterations=args.iterations,
        save_iterations=args.save_iterations,
    )
    text, meta = worker.process_file(args.image)

    with open(args.output, "w", encoding="utf-8") as fh:
        fh.write(text)

    print("--- Result ---")
    print(text)
    print("--- Meta ---")
    for k, v in meta.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
