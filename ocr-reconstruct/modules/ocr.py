"""Thin wrapper around pytesseract to centralize config and calls."""
import pytesseract
import cv2


def image_to_text(img, lang="eng", psm=6, oem=3):
    # Accept grayscale or color images
    if len(img.shape) == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img

    config = f"--oem {oem} --psm {psm}"
    text = pytesseract.image_to_string(img_gray, lang=lang, config=config)
    return text.strip()
