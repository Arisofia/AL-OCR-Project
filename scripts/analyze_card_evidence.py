#!/usr/bin/env python3
"""Analyze OCR partial reads and find Luhn-valid PAN candidates."""

import itertools
import re
import sys

sys.path.insert(0, ".")

from ocr_service.modules.personal_doc_extractor import _luhn_valid

# All partial reads containing prefix 438854 and/or suffix 0665
# from the extract_card_digits.py output
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

print("=== Analyzing partial OCR reads for digit evidence ===\n")
for p in PARTIALS:
    m = re.search(r"438854(.*)0665", p)
    if m:
        middle = m.group(1)
        print(f"  {p:20s}  middle=[{middle}]  len_middle={len(middle)}")
    else:
        m2 = re.search(r"438854(.*)", p)
        if m2:
            after = m2.group(1)
            print(f"  {p:20s}  after_prefix=[{after}]")

print("""
=== Digit evidence from OCR partial reads ===

Position 6: OCR reads suggest 0, 5, 6, 7 or 9 (from multiple PSM13 reads)
Position 7: OCR reads suggest 0 or 2 (from 43885452665 → '52' middle)
Positions 8-11: mostly blank (fully occluded by marker)
""")

PREFIX = "438854"
SUFFIX = "0665"
UNKNOWN = 6  # positions 6-11

print(f"Searching all valid PANs: {PREFIX}??????{SUFFIX}\n")

valid = []
for combo in itertools.product("0123456789", repeat=UNKNOWN):
    mid = "".join(combo)
    pan = PREFIX + mid + SUFFIX
    if _luhn_valid(pan):
        valid.append(pan)

print(f"Total Luhn-valid: {len(valid)}\n")

# Filter by OCR evidence at position 6
ocr_pos6 = set("056789")
filtered = [p for p in valid if p[6] in ocr_pos6]
print(f"Filtered by pos-6 OCR evidence [{'|'.join(sorted(ocr_pos6))}]: {len(filtered)}")

# Filter by OCR evidence at position 7
ocr_pos7 = set("02")
filtered2 = [p for p in filtered if p[7] in ocr_pos7]
print(f"  + pos-7 evidence [{'|'.join(sorted(ocr_pos7))}]: {len(filtered2)}")

print(f"\n=== Top 50 candidates (pos-6 + pos-7 filtered, Luhn-valid) ===\n")
for i, pan in enumerate(filtered2[:50], 1):
    g = f"{pan[0:4]} {pan[4:8]} {pan[8:12]} {pan[12:16]}"
    print(f"  {i:3d}. {g}")

print(f"\n  ... total: {len(filtered2)} Luhn-valid candidates")

# Now try to get more evidence from the longer partial reads
# that contain 438854 embedded in noise
print("\n=== Analysis of longer OCR reads containing '438854' ===\n")
longer_reads = [
    ("PSM11/full-image/fixed-180", "824745181204164388542457444203284528"),
    ("PSM11/full-image/fixed-100", "0434782256377043343885068240003277"),
    ("PSM11/full-image/fixed-120", "448447442438854779205303283"),
    ("PSM11/full-image/otsu", "7488238438854729433032842"),
    ("PSM11/full-image/fixed-140", "92779324254388549600328"),
    ("PSM11/full-image/fixed-160", "8432384388547744400328"),
    ("PSM4/full-image", "47784388540328"),
]

for source, read in longer_reads:
    m = re.search(r"438854(\d+)", read)
    if m:
        after = m.group(1)
        print(f"  {source:40s} → after 438854: [{after[:10]}...]" if len(after) > 10 else f"  {source:40s} → after 438854: [{after}]")

print("""
=== Cross-referencing all evidence ===

From PSM13 single-line reads (most reliable for isolated text):
  Position 6: {0, 5, 6, 7, 9} - multiple reads confirm digit at marker edge
  Position 7: {0, 2} - sparse evidence

From PSM11/PSM4 full-image reads (noisier but provide context):
  After '438854': '24574...'  → pos 6=2, pos 7=4 (noisy)
  After '438854': '77920...'  → pos 6=7, pos 7=7 (confirms 7 at pos 6)
  After '438854': '72943...'  → pos 6=7, pos 7=2 (confirms 7 at pos 6, 2 at pos 7)
  After '438854': '96003...'  → pos 6=9, pos 7=6
  After '438854': '77444...'  → pos 6=7, pos 7=7
  After '438854': '0328'      → pos 6=0, pos 7=3
""")

# Most consistent signal: position 6 = 7 (appears 3 times in full reads)
# Position 7: 2 appears in one read with 7 at pos 6
print("Most likely position 6 digit: 7 (appears in 3 independent reads)")
print("With pos 6=7: position 7 candidates from OCR: 0, 2, 7")
print()

best = [p for p in valid if p[6] == "7"]
print(f"Candidates with pos-6=7: {len(best)}")

for pos7_guess in ["2", "0", "7"]:
    subset = [p for p in best if p[7] == pos7_guess]
    print(f"  + pos-7={pos7_guess}: {len(subset)} candidates")
    for i, pan in enumerate(subset[:10], 1):
        g = f"{pan[0:4]} {pan[4:8]} {pan[8:12]} {pan[12:16]}"
        print(f"      {i:3d}. {g}")
    if len(subset) > 10:
        print(f"      ... ({len(subset)} total)")
