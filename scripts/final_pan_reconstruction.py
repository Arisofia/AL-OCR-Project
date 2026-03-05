"""Final PAN reconstruction combining per-position pixel evidence with Luhn."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Final

try:
    from ocr_service.modules.pan_candidates import generate_pan_candidates
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from ocr_service.modules.pan_candidates import generate_pan_candidates

Candidate = tuple[str, float, str]

WEIGHTS: Final[dict[int, dict[str, float]]] = {
    6: {"0": 0.30, "5": 0.25, "4": 0.20, "2": 0.15, "7": 0.10},
    7: {"4": 0.30, "7": 0.25, "3": 0.20, "2": 0.15, "8": 0.10},
    8: {"3": 0.30, "7": 0.25, "8": 0.20, "4": 0.15, "5": 0.10},
    9: {"3": 0.30, "7": 0.25, "8": 0.20, "4": 0.15, "2": 0.10},
    10: {"3": 0.30, "7": 0.20, "4": 0.20, "8": 0.15, "1": 0.15},
    11: {"5": 0.35, "3": 0.25, "6": 0.20, "8": 0.15, "0": 0.05},
}

PREFIX: Final = "438854"
SUFFIX: Final = "0665"

TOP_PER_POS: Final[dict[int, str]] = {
    6: "05427",
    7: "47328",
    8: "37845",
    9: "37842",
    10: "37418",
    11: "53680",
}


def _format_pan(pan: str) -> str:
    """Format PAN into 4-digit groups."""
    return f"{pan[:4]} {pan[4:8]} {pan[8:12]} {pan[12:16]}"


def _score_mid_digits(mid_digits: str) -> float:
    """Compute multiplicative confidence score for a hidden 6-digit sequence."""
    score = 1.0
    for index, digit in enumerate(mid_digits):
        position = 6 + index
        score *= WEIGHTS[position].get(digit, 0.01)
    return score


def _find_valid_candidates() -> list[Candidate]:
    """Generate all Luhn-valid PAN candidates ranked by confidence."""
    constraints = {
        position: {int(digit) for digit in TOP_PER_POS[position]}
        for position in range(6, 12)
    }
    pattern = PREFIX + ("X" * 6) + SUFFIX
    pans = generate_pan_candidates(pattern, constraints=constraints, enforce_luhn=True)

    valid: list[Candidate] = []
    for pan in pans:
        mid_digits = pan[6:12]
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
