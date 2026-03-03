#!/usr/bin/env python3
"""Final PAN reconstruction combining per-position pixel evidence with Luhn."""

from __future__ import annotations

import itertools
from typing import Final

Candidate = tuple[str, float, str]


def _luhn_valid(pan: str) -> bool:
    """Luhn checksum validation."""
    total = 0
    for i, ch in enumerate(reversed(pan)):
        d = int(ch)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0

# Per-position evidence — improved pipeline (template match + expanded OCR)
# Template matching (pos6→'5' best among 0,3,4,5,6,8; pos11→'3'/'8')
# OCR sweep (corrected regex, 8 enhancements, R-channel, 7 offsets)
WEIGHTS: Final[dict[int, dict[str, float]]] = {
    6: {"0": 0.30, "5": 0.25, "4": 0.20, "2": 0.15, "7": 0.10}, # edges:'0'>'5'>'4'; template:'5'; OCR:'2'
    7: {"4": 0.30, "7": 0.25, "3": 0.20, "2": 0.15, "8": 0.10}, # edges:'4'>'8'>'3'; OCR:'7'~'2'
    8: {"3": 0.30, "7": 0.25, "8": 0.20, "4": 0.15, "5": 0.10}, # edges:'3'>'8'>'4'; OCR:'7'
    9: {"3": 0.30, "7": 0.25, "8": 0.20, "4": 0.15, "2": 0.10}, # edges:'3'>'8'>'4'; OCR:'7'
    10: {"3": 0.30, "7": 0.20, "4": 0.20, "8": 0.15, "1": 0.15},# edges:'3'>'8'>'4'; OCR:'7'~'4'
    11: {"5": 0.35, "3": 0.25, "6": 0.20, "8": 0.15, "0": 0.05},# edges:'5'>'6'>'0'; template:'3'; visual:gap
}

PREFIX: Final = "438854"
SUFFIX: Final = "0665"

# Top candidates per position (improved pipeline — template + OCR)
TOP_PER_POS: Final[dict[int, str]] = {
    6: "05427",      # edges+template+OCR combined
    7: "47328",      # edges:'4'; OCR:'7'~'2'; template:N/A
    8: "37845",      # edges:'3'; OCR:'7'; template:'5'
    9: "37842",      # edges:'3'; OCR:'7'; template:N/A
    10: "37418",     # edges:'3'; OCR:'7'~'4'
    11: "53680",     # edges:'5'; template:'3'; visual: gap on right
}


def _format_pan(pan: str) -> str:
    """Format PAN into 4-digit groups."""
    return f"{pan[0:4]} {pan[4:8]} {pan[8:12]} {pan[12:16]}"


def _iter_mid_combinations() -> itertools.product:
    """Return product iterator across candidate digits per hidden position."""
    return itertools.product(*[TOP_PER_POS[pos] for pos in range(6, 12)])


def _score_mid_digits(mid_digits: str) -> float:
    """Compute multiplicative confidence score for a hidden 6-digit sequence."""
    score = 1.0
    for index, digit in enumerate(mid_digits):
        position = 6 + index
        score *= WEIGHTS[position].get(digit, 0.01)
    return score


def _find_valid_candidates() -> list[Candidate]:
    """Generate all Luhn-valid PAN candidates ranked by confidence."""
    valid: list[Candidate] = []
    for combo in _iter_mid_combinations():
        mid_digits = "".join(combo)
        pan = PREFIX + mid_digits + SUFFIX
        if not _luhn_valid(pan):
            continue
        valid.append((pan, _score_mid_digits(mid_digits), mid_digits))
    valid.sort(key=lambda row: -row[1])
    return valid


def _print_evidence() -> None:
    """Print top OCR evidence per hidden position."""
    print("Per-position OCR evidence:")
    for position in range(6, 12):
        ranked = sorted(WEIGHTS[position].items(), key=lambda row: -row[1])
        top3 = ", ".join(f"'{digit}'={weight:.0%}" for digit, weight in ranked[:3])
        print(f"  pos {position}: {top3}")
    print()


def _print_search_stats(valid: list[Candidate]) -> None:
    """Print search-space and candidate counts."""
    total_digits = sum(len(digits) for digits in TOP_PER_POS.values())
    combinations = 1
    for digits in TOP_PER_POS.values():
        combinations *= len(digits)
    print(f"Search space: {total_digits} digits across 6 positions")
    print(f"Combinations tested: {combinations:,}")
    print(f"Luhn-valid candidates: {len(valid)}\n")


def _print_ranked_results(valid: list[Candidate], limit: int = 20) -> None:
    """Print ranked candidates."""
    print("=== RANKED RESULTS (by pixel confidence) ===\n")
    for index, (pan, score, mid_digits) in enumerate(valid[:limit], 1):
        print(
            f"  {index:3d}. {_format_pan(pan)}  "
            f"score={score:.8e}  hidden=[{mid_digits}]"
        )


def _print_best_match(valid: list[Candidate]) -> None:
    """Print best candidate plus top-5 shortlist."""
    if not valid:
        return

    best_pan, best_score, best_mid = valid[0]
    print(f"\n{'=' * 60}")
    print(f">>> BEST MATCH: {_format_pan(best_pan)}")
    print(f"    Occluded digits recovered: {best_mid}")
    print(f"    Pixel confidence: {best_score:.8e}")
    print("    Luhn: VALID")
    print(f"{'=' * 60}")

    print("\nTop-5 most probable PANs:")
    for index, (pan, score, _) in enumerate(valid[:5], 1):
        print(f"  {index}. {_format_pan(pan)}  (confidence: {score:.6e})")


def main():
    """Find Luhn-valid PANs ranked by pixel confidence."""
    print("=== Pixel Evidence + Luhn Card Reconstruction ===\n")
    print("Known: 4388 54?? ???? 0665\n")
    _print_evidence()
    valid = _find_valid_candidates()
    _print_search_stats(valid)
    _print_ranked_results(valid)
    _print_best_match(valid)


if __name__ == "__main__":
    main()
