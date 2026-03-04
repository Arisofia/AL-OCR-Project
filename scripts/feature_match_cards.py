#!/usr/bin/env python3
"""
Use feature matching (ORB/SIFT) between original and new card images
to find homography and map digit positions precisely.
Then read the hidden digits from the new image.
"""
import re
from collections import Counter

import cv2
import numpy as np
import pytesseract

IMG1 = "/Users/jenineferderas/Desktop/card_image.jpg"
IMG2 = "/Users/jenineferderas/Downloads/20241007_002852000_iOS 3.jpg"
WL = "-c tessedit_char_whitelist=0123456789"

img1 = cv2.imread(IMG1)
img2 = cv2.imread(IMG2)
h1, w1 = img1.shape[:2]
h2, w2 = img2.shape[:2]
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

print(f"Image 1: {w1}x{h1}")
print(f"Image 2: {w2}x{h2}")

# ─── Step 1: Feature matching ───
print("\n=== FEATURE MATCHING ===\n")

orb = cv2.ORB_create(nfeatures=5000)
kp1, des1 = orb.detectAndCompute(gray1, None)
kp2, des2 = orb.detectAndCompute(gray2, None)
print(f"ORB keypoints: img1={len(kp1)}, img2={len(kp2)}")

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda m: m.distance)
print(f"Matches: {len(matches)} (best dist={matches[0].distance if matches else -1})")

# Show top matches
for m in matches[:10]:
    pt1 = kp1[m.queryIdx].pt
    pt2 = kp2[m.trainIdx].pt
    print(f"  ({pt1[0]:.0f},{pt1[1]:.0f}) -> ({pt2[0]:.0f},{pt2[1]:.0f})  dist={m.distance}")

# Try homography with good matches
good = matches[:100]
if len(good) >= 4:
    pts1 = np.reshape(np.float32([kp1[m.queryIdx].pt for m in good]), (-1, 1, 2))
    pts2 = np.reshape(np.float32([kp2[m.trainIdx].pt for m in good]), (-1, 1, 2))
    
    H, mask = cv2.findHomography(
        pts1,
        pts2,
        method=cv2.RANSAC,
        ransacReprojThreshold=5.0,
    )
    inliers = mask.ravel().sum()
    print(f"\nHomography inliers: {inliers}/{len(good)}")
    
    if H is not None:
        print(f"Homography matrix:\n{H}")
        
        # Map digit centers from image 1 to image 2
        ORIG_CENTERS = {
            0: 197, 1: 272, 2: 347, 3: 422,
            4: 497, 5: 572, 6: 647, 7: 722,
            8: 833, 9: 908, 10: 983, 11: 1058,
            12: 1133, 13: 1208, 14: 1283, 15: 1358,
        }
        
        # Y center of digit row in original image
        digit_y_orig = int(h1 * 0.40)  # approximately 40% from top
        
        print("\n=== MAPPED DIGIT POSITIONS ===\n")
        mapped_centers = {}
        for pos in range(16):
            cx = ORIG_CENTERS[pos]
            pt = np.float32([[[cx, digit_y_orig]]])
            mapped = cv2.perspectiveTransform(pt, H)
            mx, my = mapped[0][0]
            mapped_centers[pos] = (int(mx), int(my))
            known = "✓" if pos in {0,1,2,3,4,5,12,13,14,15} else "?"
            print(f"  pos{pos:2d}: ({cx},{digit_y_orig}) -> ({int(mx)},{int(my)}) [{known}]")
        
        # ─── Step 2: Read digits using mapped positions ───
        print("\n=== READING DIGITS AT MAPPED POSITIONS ===\n")
        
        KNOWN = {0:"4", 1:"3", 2:"8", 3:"8", 4:"5", 5:"4", 12:"0", 13:"6", 14:"6", 15:"5"}
        r2 = img2[:,:,2]  # R-channel
        
        # Try different Y offsets from mapped position
        HALF = 12  # Half-width of digit zone in new image
        
        for y_offset in range(-15, 20, 5):
            correct = 0
            total = 0
            
            for pos in sorted(KNOWN.keys()):
                expected = KNOWN[pos]
                mx, my = mapped_centers[pos]
                cy = my + y_offset
                
                # ROI around digit
                y0 = max(0, cy - 20)
                y1 = min(h2, cy + 20)
                x0 = max(0, mx - HALF)
                x1 = min(w2, mx + HALF)
                
                if x1 <= x0 or y1 <= y0:
                    continue
                
                zone_g = gray2[y0:y1, x0:x1]
                zone_r = r2[y0:y1, x0:x1]
                
                votes = Counter()
                for src in [zone_g, zone_r]:
                    up = cv2.resize(src, (src.shape[1]*6, src.shape[0]*6), interpolation=cv2.INTER_CUBIC)
                    for clip in [8, 16, 32, 64]:
                        enh = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(3,3)).apply(up)
                        inv = cv2.bitwise_not(enh)
                        for psm in [10, 13]:
                            try:
                                txt = pytesseract.image_to_string(inv, config=f"--oem 3 --psm {psm} {WL}").strip()
                                d = re.sub(r'\D', '', txt)
                                if d: votes[d[0]] += 1
                            except Exception: pass
                            # Binary threshold
                            for thr in [120, 150]:
                                try:
                                    _, bw = cv2.threshold(inv, thr, 255, cv2.THRESH_BINARY)
                                    txt = pytesseract.image_to_string(bw, config=f"--oem 3 --psm {psm} {WL}").strip()
                                    d = re.sub(r'\D', '', txt)
                                    if d: votes[d[0]] += 1
                                except Exception: pass
                
                best = votes.most_common(1)[0][0] if votes else "?"
                if best == expected: correct += 1
                total += 1
            
            pct = correct / total * 100 if total else 0
            print(f"  y_offset={y_offset:+3d}: {correct}/{total} correct ({pct:.0f}%)")
            
            if pct >= 50:
                print(f"\n  >>> USING y_offset={y_offset} <<<\n")
                
                # Read ALL positions including hidden
                print("  Known digit verification:")
                for pos in sorted(KNOWN.keys()):
                    expected = KNOWN[pos]
                    mx, my = mapped_centers[pos]
                    cy = my + y_offset
                    y0 = max(0, cy - 20)
                    y1 = min(h2, cy + 20)
                    x0 = max(0, mx - HALF)
                    x1 = min(w2, mx + HALF)
                    
                    zone_g = gray2[y0:y1, x0:x1]
                    zone_r = r2[y0:y1, x0:x1]
                    votes = Counter()
                    for src in [zone_g, zone_r]:
                        up = cv2.resize(src, (src.shape[1]*6, src.shape[0]*6), interpolation=cv2.INTER_CUBIC)
                        for clip in [8, 16, 32, 64]:
                            enh = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(3,3)).apply(up)
                            inv = cv2.bitwise_not(enh)
                            for psm in [10, 13]:
                                try:
                                    txt = pytesseract.image_to_string(inv, config=f"--oem 3 --psm {psm} {WL}").strip()
                                    d = re.sub(r'\D', '', txt)
                                    if d: votes[d[0]] += 1
                                except Exception: pass
                                for thr in [120, 150]:
                                    try:
                                        _, bw = cv2.threshold(inv, thr, 255, cv2.THRESH_BINARY)
                                        txt = pytesseract.image_to_string(bw, config=f"--oem 3 --psm {psm} {WL}").strip()
                                        d = re.sub(r'\D', '', txt)
                                        if d: votes[d[0]] += 1
                                    except Exception: pass
                    
                    total_v = sum(votes.values())
                    top3 = ", ".join(f"'{k}'={v}" for k,v in votes.most_common(3)) if total_v else "none"
                    best = votes.most_common(1)[0][0] if votes else "?"
                    hit = "✓" if best == expected else "✗"
                    print(f"    pos{pos:2d}: expect='{expected}' got='{best}' [{hit}] ({top3})")
                
                print("\n  HIDDEN POSITIONS (6-11):")
                hidden_votes = {}
                for pos in range(6, 12):
                    mx, my = mapped_centers[pos]
                    cy = my + y_offset
                    
                    votes = Counter()
                    # Try multiple x-offsets too
                    for dx in range(-4, 5, 2):
                        y0 = max(0, cy - 20)
                        y1 = min(h2, cy + 20)
                        x0 = max(0, mx + dx - HALF)
                        x1 = min(w2, mx + dx + HALF)
                        
                        zone_g = gray2[y0:y1, x0:x1]
                        zone_r = r2[y0:y1, x0:x1]
                        
                        for src in [zone_g, zone_r]:
                            up = cv2.resize(src, (src.shape[1]*6, src.shape[0]*6), interpolation=cv2.INTER_CUBIC)
                            for clip in [8, 16, 32, 64]:
                                enh = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(3,3)).apply(up)
                                inv = cv2.bitwise_not(enh)
                                for psm in [10, 13]:
                                    try:
                                        txt = pytesseract.image_to_string(inv, config=f"--oem 3 --psm {psm} {WL}").strip()
                                        d = re.sub(r'\D', '', txt)
                                        if d: votes[d[0]] += 1
                                    except Exception: pass
                                    for thr in [100, 120, 140, 160]:
                                        try:
                                            _, bw = cv2.threshold(inv, thr, 255, cv2.THRESH_BINARY)
                                            txt = pytesseract.image_to_string(bw, config=f"--oem 3 --psm {psm} {WL}").strip()
                                            d = re.sub(r'\D', '', txt)
                                            if d: votes[d[0]] += 1
                                        except Exception: pass
                            # Gamma
                            for gamma in [0.3, 0.5]:
                                lut = np.array([((i/255.0)**gamma)*255 for i in range(256)], np.uint8)
                                bright = cv2.bitwise_not(cv2.LUT(up, lut))
                                for psm in [10, 13]:
                                    try:
                                        txt = pytesseract.image_to_string(bright, config=f"--oem 3 --psm {psm} {WL}").strip()
                                        d = re.sub(r'\D', '', txt)
                                        if d: votes[d[0]] += 1
                                    except Exception: pass
                            # Tophat
                            kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
                            tophat = cv2.morphologyEx(up, cv2.MORPH_TOPHAT, kern)
                            for psm in [10, 13]:
                                try:
                                    txt = pytesseract.image_to_string(tophat, config=f"--oem 3 --psm {psm} {WL}").strip()
                                    d = re.sub(r'\D', '', txt)
                                    if d: votes[d[0]] += 1
                                except Exception: pass
                    
                    hidden_votes[pos] = votes
                    total_v = sum(votes.values())
                    top5 = ", ".join(f"'{k}'={v}" for k,v in votes.most_common(5)) if total_v else "none"
                    best = votes.most_common(1)[0][0] if votes else "?"
                    print(f"    pos{pos:2d}: BEST='{best}'  ({top5})  total={total_v}")
                
                # Save enhanced hidden zones
                for pos in range(6, 12):
                    mx, my = mapped_centers[pos]
                    cy = my + y_offset
                    y0 = max(0, cy - 20)
                    y1 = min(h2, cy + 20)
                    x0 = max(0, mx - HALF)
                    x1 = min(w2, mx + HALF)
                    zone = gray2[y0:y1, x0:x1]
                    zone_r = r2[y0:y1, x0:x1]
                    up_g = cv2.resize(zone, (zone.shape[1]*8, zone.shape[0]*8), interpolation=cv2.INTER_CUBIC)
                    up_r = cv2.resize(zone_r, (zone_r.shape[1]*8, zone_r.shape[0]*8), interpolation=cv2.INTER_CUBIC)
                    enh_g = cv2.bitwise_not(cv2.createCLAHE(clipLimit=32.0, tileGridSize=(3,3)).apply(up_g))
                    enh_r = cv2.bitwise_not(cv2.createCLAHE(clipLimit=32.0, tileGridSize=(3,3)).apply(up_r))
                    cv2.imwrite(f"/tmp/new2_pos{pos}_gray.png", enh_g)
                    cv2.imwrite(f"/tmp/new2_pos{pos}_rchan.png", enh_r)
                
                # ─── Luhn with new image evidence ───
                print("\n  LUHN CHECK on pattern 4388 5454 ???5 0665:")
                
                # Focus on positions 8, 9, 10 which are the unknowns
                def luhn(pan):
                    t = 0
                    for i, c in enumerate(reversed(pan)):
                        d = int(c)
                        if i % 2 == 1: d *= 2; d = d - 9 if d > 9 else d
                        t += d
                    return t % 10 == 0
                
                # Get top candidates for pos 8, 9, 10
                for pos in [8, 9, 10]:
                    v = hidden_votes.get(pos, Counter())
                    total_v = sum(v.values())
                    if total_v:
                        probs = ", ".join(f"'{k}'={v_/total_v:.0%}" for k, v_ in v.most_common(5))
                        print(f"    pos{pos}: {probs}")
                
                # Check all 100 Luhn-valid candidates
                print("\n  Ranking among 100 Luhn-valid candidates for 4388 5454 ???5 0665:")
                candidates = []
                for d8 in range(10):
                    for d9 in range(10):
                        for d10 in range(10):
                            pan = f"43885454{d8}{d9}{d10}50665"
                            if luhn(pan):
                                # Score from new image
                                v8 = hidden_votes.get(8, Counter())
                                v9 = hidden_votes.get(9, Counter())
                                v10 = hidden_votes.get(10, Counter())
                                t8 = sum(v8.values()) or 1
                                t9 = sum(v9.values()) or 1
                                t10 = sum(v10.values()) or 1
                                s = (v8.get(str(d8), 0)/t8) * (v9.get(str(d9), 0)/t9) * (v10.get(str(d10), 0)/t10)
                                candidates.append((pan, s))
                
                candidates.sort(key=lambda x: -x[1])
                for i, (pan, s) in enumerate(candidates[:15], 1):
                    formatted = f"{pan[:4]} {pan[4:8]} {pan[8:12]} {pan[12:16]}"
                    print(f"    {i:3d}. {formatted}  score={s:.6e}")
                
                break  # Use first good y_offset
    else:
        print("\nHomography failed!")
else:
    print("\nNot enough matches for homography!")

# ─── Also try SIFT ───
print("\n\n=== TRYING SIFT ===\n")
try:
    sift = cv2.SIFT_create(nfeatures=5000)
    kp1s, des1s = sift.detectAndCompute(gray1, None)
    kp2s, des2s = sift.detectAndCompute(gray2, None)
    print(f"SIFT keypoints: img1={len(kp1s)}, img2={len(kp2s)}")
    
    bf2 = cv2.BFMatcher(cv2.NORM_L2)
    matches_s = bf2.knnMatch(des1s, des2s, k=2)
    
    # Lowe's ratio test
    good_s = [m for m, n in matches_s if m.distance < 0.7 * n.distance]
    
    print(f"Good SIFT matches: {len(good_s)}")
    
    if len(good_s) >= 10:
        pts1s = np.reshape(np.float32([kp1s[m.queryIdx].pt for m in good_s]), (-1, 1, 2))
        pts2s = np.reshape(np.float32([kp2s[m.trainIdx].pt for m in good_s]), (-1, 1, 2))
        
        Hs, masks = cv2.findHomography(
            pts1s,
            pts2s,
            method=cv2.RANSAC,
            ransacReprojThreshold=5.0,
        )
        inliers_s = masks.ravel().sum() if masks is not None else 0
        print(f"SIFT Homography inliers: {inliers_s}/{len(good_s)}")
        
        if Hs is not None:
            # Map digit positions
            digit_y_orig = int(h1 * 0.40)
            ORIG_CENTERS = {
                0: 197, 1: 272, 2: 347, 3: 422,
                4: 497, 5: 572, 6: 647, 7: 722,
                8: 833, 9: 908, 10: 983, 11: 1058,
                12: 1133, 13: 1208, 14: 1283, 15: 1358,
            }
            
            print("\n  SIFT-mapped positions:")
            for pos in range(16):
                cx = ORIG_CENTERS[pos]
                pt = np.float32([[[cx, digit_y_orig]]])
                mapped = cv2.perspectiveTransform(pt, Hs)
                mx, my = mapped[0][0]
                known = "✓" if pos in {0,1,2,3,4,5,12,13,14,15} else "?"
                print(f"    pos{pos:2d}: -> ({int(mx)},{int(my)}) [{known}]")
except Exception as e:
    print(f"SIFT error: {e}")

print("\nDone.")
