"""Analyze OCR partial reads and find Luhn-valid PAN candidates."""

from __future__ import annotations

import itertools
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ocr_service.modules.pan_candidates import (
    generate_pan_candidates,
)


PARTIALS = [
    "43885400665",
    "43885470665",
    "43885460665",
    "43885450665",
    "43885452665",
    "4388540665",
    "4388549065",
    "4388545665",
    "438854665",
    "438854043",
]

LONGER_READS = [
    ("PSM11/full-image/fixed-180", "824745181204164388542457444203284528"),
    ("PSM11/full-image/fixed-100", "0434782256377043343885068240003277"),
    ("PSM11/full-image/fixed-120", "448447442438854779205303283"),
    ("PSM11/full-image/otsu", "7488238438854729433032842"),
    ("PSM11/full-image/fixed-140", "92779324254388549600328"),
    ("PSM11/full-image/fixed-160", "8432384388547744400328"),
    ("PSM4/full-image", "47784388540328"),
]


def analyze_partials() -> None:
    """Print OCR partial-read evidence around known prefix and suffix."""
    print("=== Analyzing partial OCR reads for digit evidence ===\n")
    for partial in PARTIALS:
        if match := re.search(r"438854(.*)0665", partial):
            middle = match[1]
            print(f"  {partial:20s}  middle=[{middle}]  len_middle={len(middle)}")
        elif match_after := re.search(r"438854(.*)", partial):
            after = match_after[1]
            print(f"  {partial:20s}  after_prefix=[{after}]")

    print(
        """
=== Digit evidence from OCR partial reads ===

Position 6: OCR reads suggest 0, 5, 6, 7 or 9 (from multiple PSM13 reads)
Position 7: OCR reads suggest 0 or 2 (from 43885452665 -> '52' middle)
Positions 8-11: mostly blank (fully occluded by marker)
"""
    )


def build_luhn_candidates(prefix: str, suffix: str, unknown: int) -> list[str]:
    """Brute-force all middle digits and keep only Luhn-valid PANs."""
    pattern = prefix + ("X" * unknown) + suffix
    return generate_pan_candidates(pattern, enforce_luhn=True)


def print_candidate_summary(valid_candidates: list[str]) -> list[str]:
    """Print filtered candidate summary and return narrowed candidates."""
    print(f"Total Luhn-valid: {len(valid_candidates)}\n")

    ocr_pos6 = set("056789")
    filtered = [candidate for candidate in valid_candidates if candidate[6] in ocr_pos6]
    print(
        f"Filtered by pos-6 OCR evidence "
        f"[{'|'.join(sorted(ocr_pos6))}]: {len(filtered)}"
    )

    ocr_pos7 = set("02")
    filtered2 = [candidate for candidate in filtered if candidate[7] in ocr_pos7]
    print(f"  + pos-7 evidence [{'|'.join(sorted(ocr_pos7))}]: {len(filtered2)}")

    print("\n=== Top 50 candidates (pos-6 + pos-7 filtered, Luhn-valid) ===\n")
    for idx, pan in enumerate(filtered2[:50], 1):
        grouped = f"{pan[:4]} {pan[4:8]} {pan[8:12]} {pan[12:16]}"
        print(f"  {idx:3d}. {grouped}")

    print(f"\n  ... total: {len(filtered2)} Luhn-valid candidates")
    return filtered2


def print_longer_read_analysis() -> None:
    """Print evidence extracted from longer OCR reads containing the prefix."""
    print("\n=== Analysis of longer OCR reads containing '438854' ===\n")
    for source, read in LONGER_READS:
        if match := re.search(r"438854(\d+)", read):
            after = match[1]
            snippet = f"{after[:10]}..." if len(after) > 10 else after
            print(f"  {source:40s} -> after 438854: [{snippet}]")

    print(
        """
=== Cross-referencing all evidence ===

From PSM13 single-line reads (most reliable for isolated text):
  Position 6: {0, 5, 6, 7, 9} - multiple reads confirm digit at marker edge
  Position 7: {0, 2} - sparse evidence

From PSM11/PSM4 full-image reads (noisier but provide context):
  After '438854': '24574...'  -> pos 6=2, pos 7=4 (noisy)
  After '438854': '77920...'  -> pos 6=7, pos 7=7 (confirms 7 at pos 6)
  After '438854': '72943...'  -> pos 6=7, pos 7=2 (confirms 7 at pos 6, 2 at pos 7)
  After '438854': '96003...'  -> pos 6=9, pos 7=6
  After '438854': '77444...'  -> pos 6=7, pos 7=7
  After '438854': '0328'      -> pos 6=0, pos 7=3
"""
    )


def print_best_slices(valid_candidates: list[str]) -> None:
    """Print best subsets after fixing likely position guesses."""
    print("Most likely position 6 digit: 7 (appears in 3 independent reads)")
    print("With pos 6=7: position 7 candidates from OCR: 0, 2, 7")
    print()

    best = [candidate for candidate in valid_candidates if candidate[6] == "7"]
    print(f"Candidates with pos-6=7: {len(best)}")

    for pos7_guess in ["2", "0", "7"]:
        subset = [candidate for candidate in best if candidate[7] == pos7_guess]
        print(f"  + pos-7={pos7_guess}: {len(subset)} candidates")
        for idx, pan in enumerate(subset[:10], 1):
            grouped = f"{pan[:4]} {pan[4:8]} {pan[8:12]} {pan[12:16]}"
            print(f"      {idx:3d}. {grouped}")
        if len(subset) > 10:
            print(f"      ... ({len(subset)} total)")


def main() -> None:
    """Run complete OCR evidence analysis workflow."""
    prefix = "438854"
    suffix = "0665"
    unknown = 6

    analyze_partials()
    print(f"Searching all valid PANs: {prefix}??????{suffix}\n")

    valid_candidates = build_luhn_candidates(prefix, suffix, unknown)
    filtered_candidates = print_candidate_summary(valid_candidates)
    print_longer_read_analysis()
    print_best_slices(filtered_candidates)


if __name__ == "__main__":
    main()
