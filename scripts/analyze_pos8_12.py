"""Analyze the OCR reads captured from minimal_pos8_12.py run."""

FULL_READS = [
    "4388540665",
    "4388540665",
    "4388540665",
    "4388540665",
    "438854290665",
    "43885420665",
    "4388546",
    "43885426",
    "438854665",
    "4388546658",
]

SUFFIX_READS = [
    "0665",
    "0665",
    "0665",
    "2665",
]

PREV_PSM13 = [
    "43885400665",
    "43885470665",
    "43885460665",
    "43885450665",
    "43885452665",
]

PREV_EVIDENCE = {
    6:  {"4": 0.87, "7": 0.05, "5": 0.05, "1": 0.02, "6": 0.01},
    7:  {"8": 0.47, "3": 0.43, "2": 0.08, "4": 0.02},
    8:  {"5": 0.48, "8": 0.41, "3": 0.04, "2": 0.03, "7": 0.03},
    9:  {"4": 0.65, "0": 0.23, "7": 0.07, "1": 0.03, "2": 0.01},
    10: {"7": 0.47, "5": 0.27, "0": 0.13, "2": 0.07, "4": 0.07},
    11: {"4": 0.42, "7": 0.38, "1": 0.12, "5": 0.04, "8": 0.04},
}

print("=" * 60)
print("POSITION 12 ANALYSIS")
print("=" * 60)

print("\n1. Suffix zone reads (isolated right portion):")
for r in SUFFIX_READS:
    print(f"   '{r}'")
print("   → 3/4 say '0665', 1/4 says '2665' (tophat)")

print("\n2. Full-row reads — what precedes '665'?")
for r in FULL_READS:
    idx = r.rfind("665")
    if idx >= 1:
        before = r[idx-1]
        context = r[max(0,idx-3):idx+3]
        print(f"   '{r}' → before 665: '{before}'  context: ...{context}...")
    else:
        print(f"   '{r}' → no 665 found")

print("\n3. Previous PSM13 reads — what precedes '665'?")
for r in PREV_PSM13:
    idx = r.rfind("665")
    if idx >= 1:
        before = r[idx-1]
        print(f"   '{r}' → before 665: '{before}'")

from collections import Counter
votes12 = Counter()
for r in FULL_READS + SUFFIX_READS + PREV_PSM13:
    idx = r.rfind("665")
    if idx >= 1:
        votes12[r[idx-1]] += 1

print("\n4. CONSOLIDATED VOTE for POS 12:")
total = sum(votes12.values())
for d, n in votes12.most_common():
    bar = "█" * max(1, int(n / total * 40))
    print(f"   '{d}': {n}/{total} = {n/total:.0%}  {bar}")

print("\n5. KEY OBSERVATION:")
print("   - g03i reads: '438854665' and '4388546658'")
print("     These SKIP position 12 entirely, going from prefix right to '665'")
print("     This could mean position 12 is actually '0' (OCR sees as space/nothing)")
print("     OR it could be very faint/ambiguous")
print()
print("   - tophat reads: '2665' as suffix — consistently sees '2' instead of '0'")
print("     Tophat enhances LOCAL bright features → could reveal embossed texture")
print("     of a digit that CLAHE averages out to look like '0'")
print()
print("   - Previous read '43885452665' shows '52' then '665' (no leading 0!)")
print("     This is evidence that position 12 might be '2', not '0'")
print("     Interpretation: ...??52 2665 — positions 10-11 are '52', pos 12 is '2'")

print("\n" + "=" * 60)
print("POSITION 8 ANALYSIS")
print("=" * 60)

print("\n1. Full-row reads — hidden digits between prefix and suffix:")
for r in FULL_READS:
    pi = r.find("438854")
    si = r.rfind("665")
    if pi >= 0 and si > pi:
        hidden = r[pi+6:si]
        if hidden.endswith("0") and r[si-1:si+3] == "0665":
            pre_suffix = hidden[:-1]
            suffix_start = "0665"
        else:
            pre_suffix = hidden
            suffix_start = "665"
        print(f"   '{r}' → hidden='{hidden}', pre-suffix='{pre_suffix}', suffix='{suffix_start}'")

print("\n2. Previous PSM13 reads — hidden digits:")
for r in PREV_PSM13:
    pi = r.find("438854")
    si = r.rfind("665")
    if pi >= 0 and si > pi:
        hidden = r[pi+6:si]
        print(f"   '{r}' → between prefix and '665': '{hidden}'")

print("\n3. All hidden-digit fragments collected:")
fragments = []
for r in FULL_READS + PREV_PSM13:
    pi = r.find("438854")
    si = r.rfind("665")
    if pi >= 0 and si > pi:
        h = r[pi+6:si]
        if h:
            fragments.append(h)
            
frag_counter = Counter(fragments)
for f, n in frag_counter.most_common():
    print(f"   '{f}' x{n}")

print("\n4. Key digit patterns in hidden zone:")
print("   Fragments seen: " + ", ".join(f"'{f}'" for f in sorted(set(fragments))))
print()
print("   Digit '2' appears consistently:")
print("     - '29' in c64i/x2 full-row read")
print("     - '2' in c64i/x3 full-row read")
print("     - '26' in tophat/x3 full-row read") 
print("     - '52' in previous PSM13 read")
print("   → '2' is a strong signal from the hidden zone")
print()
print("   Previous per-position evidence for POS 8:")
print("     '5'=48%, '8'=41%, '3'=4%, '2'=3%, '7'=3%")
print("     '2' was at only 3% — BUT that was with potentially wrong zone positions!")

print("\n5. Possible interpretations:")
print("   If the hidden digits are ??2??? and suffix is 0665:")
print("     The '2' could be at pos 8, 9, 10, or 11")
print("   If the hidden digits are ????52 and suffix is 2665:")
print("     pos 10='5', pos 11='2', pos 12='2' (not '0'!)")
print()
print("   Looking at '29' fragment: if these are pos 8-9 → pos8='2', pos9='9'")
print("   Looking at '26' fragment: if these are pos 8-9 → pos8='2', pos9='6'")
print("   The '2' at pos 8 would be a NEW finding vs prev '5'/'8' evidence")

print("\n" + "=" * 60)
print("REVISED HYPOTHESIS")
print("=" * 60)
print()
print("  Original evidence:   4388 54?? ???? 0665")
print("  Previous best guess: 4388 5443 5474 0665")
print()
print("  NEW evidence suggests:")
print("  A) Position 12 is likely '0' (strong: 75% of reads)")
print("     BUT tophat consistently sees '2' → worth checking '2665' variant")
print()
print("  B) Position 8 is contested:")
print("     - Boundary zone OCR: '5' (48%) vs '8' (41%)")
print("     - Full-row hidden fragments: '2' appears in 3 independent reads")
print("     - Could be '2' instead of '5' or '8'")
print()
print("  Re-ranking needed with these two scenarios:")
print("  Scenario 1: suffix = 0665, pos8 uncertain (5/8/2)")
print("  Scenario 2: suffix = 2665, different Luhn constraints")

def luhn(pan):
    t = 0
    for i, c in enumerate(reversed(pan)):
        d = int(c)
        if i % 2 == 1:
            d *= 2
            if d > 9: d -= 9
        t += d
    return t % 10 == 0

import itertools

print("\n" + "=" * 60)
print("LUHN RECOMPUTE — BOTH SUFFIX SCENARIOS")
print("=" * 60)

UPDATED = {
    6: {"4": 0.80, "7": 0.05, "5": 0.05, "2": 0.05, "6": 0.03, "1": 0.02},
    7: {"8": 0.40, "3": 0.38, "2": 0.10, "9": 0.05, "6": 0.05, "4": 0.02},
    8: {"5": 0.35, "8": 0.30, "2": 0.20, "3": 0.05, "7": 0.05, "9": 0.03, "6": 0.02},
    9: {"4": 0.50, "0": 0.20, "9": 0.10, "7": 0.07, "6": 0.05, "2": 0.05, "1": 0.03},
    10: {"7": 0.40, "5": 0.25, "0": 0.13, "2": 0.10, "4": 0.07, "6": 0.05},
    11: {"4": 0.35, "7": 0.33, "2": 0.12, "1": 0.10, "5": 0.05, "8": 0.05},
}

for suffix, suffix_label in [("0665", "SCENARIO A (suffix=0665)"), ("2665", "SCENARIO B (suffix=2665)")]:
    print(f"\n--- {suffix_label} ---")
    prefix = "438854"
    top_per_pos = {}
    for pos in range(6, 12):
        sorted_d = sorted(UPDATED[pos].items(), key=lambda x: -x[1])[:4]
        top_per_pos[pos] = "".join(d for d, _ in sorted_d)

    valid = []
    for combo in itertools.product(*[top_per_pos[p] for p in range(6, 12)]):
        mid = "".join(combo)
        pan = prefix + mid + suffix
        if len(pan) == 16 and luhn(pan):
            score = 1.0
            for i, d in enumerate(mid):
                pos = 6 + i
                score *= UPDATED[pos].get(d, 0.01)
            valid.append((pan, score, mid))

    valid.sort(key=lambda x: -x[1])
    
    combos = 1
    for v in top_per_pos.values():
        combos *= len(v)
    print(f"  Combos: {combos}, Luhn-valid: {len(valid)}")
    
    for i, (pan, score, mid) in enumerate(valid[:10], 1):
        g = f"{pan[0:4]} {pan[4:8]} {pan[8:12]} {pan[12:16]}"
        print(f"  {i:3d}. {g}  hidden=[{mid}]  score={score:.6e}")
