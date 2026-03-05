"""Final analysis: combine all evidence and produce ranked Luhn-valid PANs."""

import itertools
from collections import Counter

hits = [
    "292251224634388548403283",
    "031414654438890020328403",
    "7429633276643885420328",
    "55241876654388540328",
    "47352152344443885466810328",
    "17424438842433424542",
    "284388541100220",
    "54388540664203282",
    "5454438854066503282",
    "2538544438854275037",
    "92741425643885440328",
    "12122814977454243454388000328482",
    "51520545228404388542338",
    "55024583885406651221121",
    "7372887259407443885414000017288",
    "357427236654388544032803",
    "55352232466543885444303281",
    "34372234844745421424388540328230",
    "285411413357438854474240328",
    "7515443664388547340328",
    "5375725512747414577164527482446438854803282",
    "7442727043885442",
    "444974064388540328275",
    "24208520664388542032871",
    "44475406653228",
    "237982123457164205765438854743203282",
    "53845722835554388520328449",
    "24736643885487470328412",
    "2554444388544703281",
    "33533015713826253541873420571643885474303283",
    "2222777247433434388540665170743",
    "214226491034134388547145",
    "305842249269438854774441",
    "38142422433227143885470328",
    "64574479426643885471220328434",
    "27248477514438854220328",
    "554411777168474434249643885440328402",
    "55254722206674388544032874",
    "747097735338443438854710003985",
    "744474388540050481774037282",
    "74245548755446438854880328",
    "667152423874437946624388540328333",
    "554247368266438854303287",
    "201044242438854428440328",
    "030273654842407438854000264944113648",
    "44522445223447438854403284",
    "45297452441814166438854743203284204",
    "272788629145724388542032843",
    "7716454442443885424340303284",
    "245243477066438854040328422",
    "5542066553328440424",
    "137934438854032345",
    "5549433745322545752454743884772482827248042",
    "964456738774719167066528547",
    "4743755272164457545243885480328",
    "62225249977214514992337438855410070328747",
    "47438825014427242829120944468143323",
    "142638471466438854447427032874",
    "735429771247466443885454920328411",
    "247168283443885473603287",
]

print("=" * 60)
print("PART 1: Statistical analysis of new image OCR reads")
print("=" * 60)

after_438854 = []
for s in hits:
    idx = s.find("438854")
    if idx >= 0:
        after = s[idx + 6:]
        after_438854.append(after)

print(f"\nFound {len(after_438854)} reads containing '438854'")

print("\n--- First digit after '438854' ---")
d1 = Counter()
for a in after_438854:
    if len(a) >= 1:
        d1[a[0]] += 1
for d, c in d1.most_common():
    print(f"  '{d}': {c} ({100*c/sum(d1.values()):.0f}%)")

print("\n--- Reads containing BOTH '438854' AND '0665' ---")
both_count = 0
for s in hits:
    if "438854" in s and "0665" in s:
        both_count += 1
        i1 = s.find("438854")
        i2 = s.find("0665")
        between = s[i1 + 6 : i2]
        print(f"  between='{between}' | full: ...{s[max(0,i1-2):i2+8]}...")
print(f"  Total: {both_count} reads")

print("\n--- Digits before '0665' ---")
for s in hits:
    idx = s.find("0665")
    if idx >= 0 and idx >= 4:
        quad = s[idx - 4 : idx]
        print(f"  ...{quad}0665... from '{s}'")

print("\n--- Digit frequency at offsets after '438854' ---")
for pos in range(8):
    freq = Counter()
    for a in after_438854:
        if len(a) > pos:
            freq[a[pos]] += 1
    top = freq.most_common(5)
    print(f"  offset+{pos}: {top}")

print("\n--- First 10 chars after '438854' (sorted) ---")
for a in sorted(after_438854):
    print(f"  '{a[:12]}'")

print("\n" + "=" * 60)
print("PART 2: Luhn validation for 4388 5454 ???5 0665")
print("=" * 60)

prefix = "43885454"
suffix = "50665"

valid = []
for d8, d9, d10 in itertools.product(range(10), repeat=3):
    pan = prefix + str(d8) + str(d9) + str(d10) + suffix
    digits = [int(c) for c in pan]
    total = 0
    for i, d in enumerate(digits):
        if i % 2 == 0:
            dd = d * 2
            if dd > 9:
                dd -= 9
            total += dd
        else:
            total += d
    if total % 10 == 0:
        valid.append(pan)

print(f"\nTotal Luhn-valid PANs: {len(valid)}")

print("\n" + "=" * 60)
print("PART 3: Score and rank candidates using combined evidence")
print("=" * 60)


edge_scores = {
    8: {'3': 0.35, '7': 0.25, '8': 0.10, '0': 0.05, '4': 0.05, '5': 0.05, '1': 0.03, '2': 0.03, '6': 0.03, '9': 0.03},
    9: {'3': 0.35, '7': 0.25, '8': 0.10, '0': 0.05, '4': 0.05, '5': 0.05, '1': 0.03, '2': 0.03, '6': 0.03, '9': 0.03},
    10: {'3': 0.35, '7': 0.25, '4': 0.10, '8': 0.08, '0': 0.05, '5': 0.05, '1': 0.03, '2': 0.03, '6': 0.03, '9': 0.03},
}

scored = []
for pan in valid:
    d8, d9, d10 = pan[8], pan[9], pan[10]
    s8 = edge_scores[8].get(d8, 0.01)
    s9 = edge_scores[9].get(d9, 0.01)
    s10 = edge_scores[10].get(d10, 0.01)
    score = s8 * s9 * s10
    scored.append((pan, score, d8, d9, d10))

scored.sort(key=lambda x: -x[1])

print("\nTop 20 candidates (ranked by combined evidence score):\n")
print(f"{'Rank':<5} {'PAN':^22} {'pos8':>5} {'pos9':>5} {'pos10':>6} {'Score':>10}")
print("-" * 55)
for i, (pan, score, d8, d9, d10) in enumerate(scored[:20], 1):
    formatted = f"{pan[:4]} {pan[4:8]} {pan[8:12]} {pan[12:16]}"
    print(f"{i:<5} {formatted:^22} {d8:>5} {d9:>5} {d10:>6} {score:>10.6f}")

print(f"\nAll {len(valid)} Luhn-valid candidates:")
print("-" * 55)
for i, (pan, score, d8, d9, d10) in enumerate(scored, 1):
    formatted = f"{pan[:4]} {pan[4:8]} {pan[8:12]} {pan[12:16]}"
    print(f"{i:<5} {formatted:^22} {score:.6f}")
