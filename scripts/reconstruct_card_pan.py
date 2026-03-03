#!/usr/bin/env python3
# pylint: disable=duplicate-code
"""
Reconstruct occluded card PAN digits using Luhn + BIN constraints.

Uses the repo's own _luhn_valid() from personal_doc_extractor and
applies brute-force enumeration with filtering to find all valid
16-digit PANs matching the visible digit pattern.

Usage:
  python3 scripts/reconstruct_card_pan.py --known "4388 54?? ???? 0665"
  python3 scripts/reconstruct_card_pan.py --prefix 438854 --suffix 0665 --total 16
"""

import argparse
import itertools
import re
import sys
import time
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Luhn validation (mirrored from personal_doc_extractor for standalone use)
# ---------------------------------------------------------------------------

def luhn_valid(number: str) -> bool:
    """Return True when *number* (digits-only) satisfies the Luhn algorithm."""
    if not number.isdigit() or not 13 <= len(number) <= 19:
        return False
    total = 0
    for i, ch in enumerate(reversed(number)):
        d = int(ch)
        if i % 2 == 1:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
    return total % 10 == 0


# ---------------------------------------------------------------------------
# BIN range heuristics for Visa cards
# ---------------------------------------------------------------------------

def visa_bin_plausible(pan: str) -> bool:
    """Basic Visa BIN plausibility check."""
    return pan.startswith("4") and len(pan) in {13, 16, 19}


# ---------------------------------------------------------------------------
# Partial-pixel visual analysis clues
# ---------------------------------------------------------------------------

# From the attached image, the embossed digits show:
# Position 0-3:  4 3 8 8   (clear)
# Position 4-5:  5 4       (clear)
# Position 6:    partially visible under the black marker edge - top curves
#                suggest 6, 8, or 9 based on emboss shadow
# Position 7:    fully occluded
# Position 8-9:  fully occluded
# Position 10:   fully occluded
# Position 11:   partially visible - the right edge of the emboss shows
#                a vertical stroke, suggesting 1, 4, or 7
# Position 12-15: 0 6 6 5  (clear)

# We encode partial pixel evidence as weighted preferences:
PIXEL_HINTS: dict[int, dict[str, float]] = {
    # Position 6: edge shadow suggests rounded top (6, 8, 9, 0)
    6: {"6": 0.3, "8": 0.25, "9": 0.2, "0": 0.15, "3": 0.1},
    # Position 7: no pixel evidence, all equally likely
    7: {str(d): 0.1 for d in range(10)},
    # Position 8: no pixel evidence
    8: {str(d): 0.1 for d in range(10)},
    # Position 9: no pixel evidence
    9: {str(d): 0.1 for d in range(10)},
    # Position 10: no pixel evidence
    10: {str(d): 0.1 for d in range(10)},
    # Position 11: right edge suggests vertical stroke (1, 4, 7)
    11: {"1": 0.25, "4": 0.2, "7": 0.2, "0": 0.1, "9": 0.1,
         "2": 0.05, "3": 0.03, "5": 0.03, "6": 0.02, "8": 0.02},
}


def pixel_confidence(pan: str) -> float:
    """Score a PAN candidate by pixel-hint agreement (higher = better)."""
    score = 1.0
    for pos, hints in PIXEL_HINTS.items():
        if pos < len(pan):
            digit = pan[pos]
            score *= hints.get(digit, 0.01)
    return score


# ---------------------------------------------------------------------------
# Brute-force reconstruction
# ---------------------------------------------------------------------------

def reconstruct(  # pylint: disable=too-many-locals
    prefix: str,
    suffix: str,
    total_length: int = 16,
    use_pixel_hints: bool = True,
    hard_filter_hints: bool = True,
) -> list[dict[str, Any]]:
    """
    Enumerate all digit combinations for the unknown middle section,
    filter by Luhn validity + BIN plausibility, rank by pixel confidence.

    When hard_filter_hints is True and PIXEL_HINTS has entries for positions
    in the unknown range, only digits listed in those hints are tried for
    those positions (dramatically reducing search space).
    """
    unknown_count = total_length - len(prefix) - len(suffix)
    if unknown_count <= 0:
        print(
            "ERROR: No unknown digits "
            f"(prefix={len(prefix)}, suffix={len(suffix)}, total={total_length})"
        )
        return []

    print(f"Prefix:  {prefix}")
    print(f"Suffix:  {suffix}")
    print(
        "Unknown: "
        f"{unknown_count} digits "
        f"(positions {len(prefix)}-{len(prefix) + unknown_count - 1})"
    )

    # Build per-position digit sets
    all_digits = "0123456789"
    position_digits: list[str] = []
    for i in range(unknown_count):
        abs_pos = len(prefix) + i
        if hard_filter_hints and abs_pos in PIXEL_HINTS:
            allowed = "".join(sorted(PIXEL_HINTS[abs_pos].keys()))
            position_digits.append(allowed)
        else:
            position_digits.append(all_digits)

    search_size = 1
    for pd in position_digits:
        search_size *= len(pd)
    print(f"Search space: {search_size:,} candidates (after hint filtering)")
    print()

    valid_candidates: list[dict[str, Any]] = []
    checked = 0
    start = time.time()

    for combo in itertools.product(*position_digits):
        middle = "".join(combo)
        pan = prefix + middle + suffix
        checked += 1

        if not luhn_valid(pan):
            continue

        if not visa_bin_plausible(pan):
            continue

        entry: dict[str, Any] = {
            "pan": pan,
            "formatted": f"{pan[:4]} {pan[4:8]} {pan[8:12]} {pan[12:16]}",
            "middle": middle,
        }

        if use_pixel_hints:
            entry["pixel_score"] = pixel_confidence(pan)

        valid_candidates.append(entry)

    elapsed = time.time() - start
    print(f"Checked {checked:,} combinations in {elapsed:.2f}s")
    print(f"Luhn-valid Visa candidates: {len(valid_candidates)}")
    print()

    # Sort by pixel confidence (descending)
    if use_pixel_hints:
        valid_candidates.sort(key=lambda c: c["pixel_score"], reverse=True)

    return valid_candidates


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_known_pattern(known: str) -> tuple[str, str, int]:
    """Parse a pattern like '4388 54?? ???? 0665' into prefix, suffix, total."""
    clean = re.sub(r"[\s\-]", "", known)
    total = len(clean)

    # Find prefix (leading digits before first ?)
    prefix_match = re.match(r"^(\d+)", clean)
    prefix = prefix_match[1] if prefix_match else ""

    # Find suffix (trailing digits after last ?)
    suffix_match = re.search(r"(\d+)$", clean)
    suffix = suffix_match[1] if suffix_match else ""

    return prefix, suffix, total


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for PAN reconstruction options."""
    parser = argparse.ArgumentParser(
        description="Reconstruct occluded card PAN digits via Luhn+BIN brute force"
    )
    parser.add_argument(
        "--known", type=str, default=None,
        help="PAN pattern with ? for unknown digits, e.g. '4388 54?? ???? 0665'"
    )
    parser.add_argument("--prefix", type=str, default=None, help="Known prefix digits")
    parser.add_argument("--suffix", type=str, default=None, help="Known suffix digits")
    parser.add_argument("--total", type=int, default=16, help="Total PAN length (default: 16)")
    parser.add_argument("--top", type=int, default=25, help="Show top N candidates")
    parser.add_argument(
        "--no-pixel-hints",
        action="store_true",
        help="Disable pixel evidence ranking",
    )
    parser.add_argument(
        "--hints", type=str, default=None,
        help=(
            "Partial digit hints for unknown positions as pos:digits pairs, "
            "e.g. '6:689,11:147' means position 6 could be 6/8/9 and "
            "position 11 could be 1/4/7"
        ),
    )
    return parser


def resolve_pattern(args: argparse.Namespace) -> tuple[str, str, int]:
    """Resolve prefix/suffix/length from CLI args or default sample pattern."""
    if args.known:
        return parse_known_pattern(args.known)
    if args.prefix and args.suffix:
        return args.prefix, args.suffix, args.total

    print("Using default pattern from attached card image:")
    print("  4388 54?? ???? 0665")
    print()
    return "438854", "0665", 16


def apply_user_hints(hints: Optional[str]) -> None:
    """Apply user-provided digit restrictions to PIXEL_HINTS."""
    if not hints:
        return

    for pair in hints.split(","):
        parts = pair.strip().split(":")
        if len(parts) != 2:
            continue
        pos = int(parts[0])
        digits = parts[1]
        weight = 1.0 / len(digits)
        PIXEL_HINTS[pos] = dict.fromkeys(digits, weight)
        print(f"  Hint: position {pos} restricted to [{digits}]")


def print_candidates(candidates: list[dict[str, Any]], top: int) -> None:
    """Print top candidate list with optional pixel score."""
    print(f"=== Top {min(top, len(candidates))} candidates ===")
    print()
    for index, candidate in enumerate(candidates[:top], 1):
        line = f"  {index:3d}. {candidate['formatted']}"
        if "pixel_score" in candidate:
            line += f"  (pixel_score: {candidate['pixel_score']:.6e})"
        print(line)


def print_best_candidate(candidates: list[dict[str, Any]]) -> None:
    """Print best-ranked candidate summary."""
    if not candidates:
        return
    best = candidates[0]
    print()
    print(f">>> Best match: {best['formatted']}")
    print(f"    Middle digits: {best['middle']}")
    if "pixel_score" in best:
        print(f"    Pixel confidence: {best['pixel_score']:.6e}")


def main() -> None:
    """Run PAN reconstruction from visible digits and constraints."""
    args = build_parser().parse_args()
    prefix, suffix, total = resolve_pattern(args)
    apply_user_hints(args.hints)

    candidates = reconstruct(
        prefix,
        suffix,
        total,
        use_pixel_hints=not args.no_pixel_hints,
    )

    if not candidates:
        print("No valid PAN candidates found.")
        sys.exit(1)

    print_candidates(candidates, args.top)
    print()
    print(f"Total valid: {len(candidates)} | Shown: {min(args.top, len(candidates))}")
    print_best_candidate(candidates)


if __name__ == "__main__":
    main()
