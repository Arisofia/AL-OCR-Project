"""Cross-validate top PAN candidates against partial OCR reads."""

PARTIAL_READS = [
    "43885400665",
    "43885470665",
    "43885460665",
    "43885450665",
    "43885452665",
]

TOP_CANDIDATES = [
    "4388544354740665",
    "4388544884570665",
    "4388544854040665",
    "4388544880770665",
    "4388544850570665",
]


def subsequence_match(sub: str, full: str) -> bool:
    """Check if sub is a subsequence of full (digits in order, not necessarily contiguous)."""
    it = iter(full)
    return all(c in it for c in sub)


def contiguous_overlap(sub: str, full: str) -> int:
    """Find longest contiguous overlap of sub that appears in full."""
    best = 0
    for i in range(len(sub)):
        for j in range(i + 1, len(sub) + 1):
            if sub[i:j] in full and (j - i) > best:
                best = j - i
    return best


print("=== Cross-Validation: Partial Reads vs Top Candidates ===\n")
for cand in TOP_CANDIDATES:
    g = f"{cand[0:4]} {cand[4:8]} {cand[8:12]} {cand[12:16]}"
    print(f"Candidate: {g}")
    for pr in PARTIAL_READS:
        is_sub = subsequence_match(pr, cand)
        overlap = contiguous_overlap(pr, cand)
        marker = " <-- subsequence match" if is_sub else ""
        print(f"  vs {pr}: subseq={is_sub}, longest_contig_overlap={overlap}{marker}")
    print()
