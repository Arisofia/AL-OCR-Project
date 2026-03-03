#!/usr/bin/env python3
"""Final PAN reconstruction combining per-position pixel evidence with Luhn."""

import itertools


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

# Per-position evidence from boundary zone OCR (vote percentages)
WEIGHTS = {
    6: {"4": 0.87, "7": 0.05, "5": 0.05, "1": 0.02, "6": 0.01},
    7: {"8": 0.47, "3": 0.43, "2": 0.08, "4": 0.02},
    8: {"5": 0.48, "8": 0.41, "3": 0.04, "2": 0.03, "7": 0.03},
    9: {"4": 0.65, "0": 0.23, "7": 0.07, "1": 0.03, "2": 0.01},
    10: {"7": 0.47, "5": 0.27, "0": 0.13, "2": 0.07, "4": 0.07},
    11: {"4": 0.42, "7": 0.38, "1": 0.12, "5": 0.04, "8": 0.04},
}

PREFIX = "438854"
SUFFIX = "0665"

# Top candidates per position (from pixel evidence)
TOP_PER_POS = {
    6: "4753",
    7: "832",
    8: "583",
    9: "407",
    10: "7504",
    11: "4718",
}


def main():
    """Find Luhn-valid PANs ranked by pixel confidence."""
    print("=== Pixel Evidence + Luhn Card Reconstruction ===\n")
    print("Known: 4388 54?? ???? 0665\n")
    print("Per-position OCR evidence:")
    for pos in range(6, 12):
        sorted_w = sorted(WEIGHTS[pos].items(), key=lambda x: -x[1])
        top3 = ", ".join(f"'{d}'={w:.0%}" for d, w in sorted_w[:3])
        print(f"  pos {pos}: {top3}")
    print()

    # Brute force with top candidates
    valid = []
    for combo in itertools.product(*[TOP_PER_POS[p] for p in range(6, 12)]):
        mid = "".join(combo)
        pan = PREFIX + mid + SUFFIX
        if _luhn_valid(pan):
            score = 1.0
            for i, d in enumerate(mid):
                pos = 6 + i
                score *= WEIGHTS[pos].get(d, 0.01)
            valid.append((pan, score, mid))

    valid.sort(key=lambda x: -x[1])

    print(f"Search space: {sum(len(v) for v in TOP_PER_POS.values())} digits across 6 positions")
    combos = 1
    for v in TOP_PER_POS.values():
        combos *= len(v)
    print(f"Combinations tested: {combos:,}")
    print(f"Luhn-valid candidates: {len(valid)}\n")

    print("=== RANKED RESULTS (by pixel confidence) ===\n")
    for i, (pan, score, mid) in enumerate(valid[:20], 1):
        g = f"{pan[0:4]} {pan[4:8]} {pan[8:12]} {pan[12:16]}"
        print(f"  {i:3d}. {g}  score={score:.8e}  hidden=[{mid}]")

    if valid:
        best_pan, best_score, best_mid = valid[0]
        g = f"{best_pan[0:4]} {best_pan[4:8]} {best_pan[8:12]} {best_pan[12:16]}"
        print(f"\n{'=' * 60}")
        print(f">>> BEST MATCH: {g}")
        print(f"    Occluded digits recovered: {best_mid}")
        print(f"    Pixel confidence: {best_score:.8e}")
        print("    Luhn: VALID")
        print(f"{'=' * 60}")

        print("\nTop-5 most probable PANs:")
        for i, (pan, score, mid) in enumerate(valid[:5], 1):
            g = f"{pan[0:4]} {pan[4:8]} {pan[8:12]} {pan[12:16]}"
            print(f"  {i}. {g}  (confidence: {score:.6e})")


if __name__ == "__main__":
    main()
