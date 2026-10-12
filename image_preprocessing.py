"""
image_preprocessing.py
Robust orientation detection + image cleanup before OCR.

Failure modes fixed:
  1. out_path uses replace('.jpg') — breaks for .jpeg, .png files → use proper path splitting
  2. No fallback if Tesseract not installed → graceful skip
  3. No EXIF orientation stripping → some phones embed rotation in EXIF
  4. No image normalisation before Tesseract scoring → dark/noisy images score poorly
  5. No maximum size cap before Tesseract → huge images are very slow
  6. Score always 0 if every conf == -1 (Tesseract returns -1 for empty) → filter -1
"""

import cv2
import numpy as np
import os

try:
    import pytesseract
    TESSERACT_OK = True
except ImportError:
    TESSERACT_OK = False

try:
    from PIL import Image as PILImage
    import PIL.ExifTags
    EXIF_OK = True
except ImportError:
    EXIF_OK = False


def _strip_exif_orientation(img_cv, image_path):
    """
    Read EXIF orientation tag and pre-rotate so we start from the correct
    physical orientation (before our own 4-way score search).
    """
    if not EXIF_OK:
        return img_cv
    try:
        pil = PILImage.open(image_path)
        exif = pil._getexif()
        if exif is None:
            return img_cv
        for tag, val in exif.items():
            if PIL.ExifTags.TAGS.get(tag) == 'Orientation':
                if val == 3:
                    return cv2.rotate(img_cv, cv2.ROTATE_180)
                elif val == 6:
                    return cv2.rotate(img_cv, cv2.ROTATE_90_CLOCKWISE)
                elif val == 8:
                    return cv2.rotate(img_cv, cv2.ROTATE_90_COUNTERCLOCKWISE)
    except Exception:
        pass
    return img_cv


def _preprocess_for_scoring(img_cv):
    """
    Lightweight normalization to make Tesseract scoring more reliable.
    - Resize to max 1200px (fast Tesseract pass)
    - Convert to gray, apply CLAHE (contrast normalisation)
    - Threshold (Otsu)
    """
    h, w = img_cv.shape[:2]
    scale = min(1200 / max(h, w, 1), 1.0)
    if scale < 1.0:
        img_cv = cv2.resize(img_cv, (int(w * scale), int(h * scale)))
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


def get_orientation_score(image_cv):
    """Score how 'upright' an image is using Tesseract confidence."""
    if not TESSERACT_OK:
        return 0

    prepped = _preprocess_for_scoring(image_cv)
    try:
        data = pytesseract.image_to_data(
            prepped, config='--psm 6 -l eng', output_type=pytesseract.Output.DICT
        )
    except Exception:
        return 0

    valid_text_count = 0
    total_conf = 0.0

    for i, text in enumerate(data['text']):
        text = text.strip()
        conf = float(data['conf'][i])
        if conf < 0:          # Tesseract returns -1 for empty segments
            continue
        if len(text) > 2 and conf > 40:
            valid_text_count += 1
            total_conf += conf

    return valid_text_count * total_conf


def find_best_orientation(image_path):
    """
    Try 0°/90°/180°/270°, score each with Tesseract, save and return
    the best-rotated image path.

    Failure modes fixed:
      • Works for .jpeg, .png, .jpg (no hard-coded suffix)
      • Saves to the uploads folder alongside source, not cwd
      • If Tesseract unavailable, defaults to angle=0
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    # Strip EXIF rotation so we start from physical orientation
    img = _strip_exif_orientation(img, image_path)

    best_score = -1
    best_angle = 0
    best_img = img

    for angle in [0, 90, 180, 270]:
        if angle == 0:
            rotated = img
        elif angle == 90:
            rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            rotated = cv2.rotate(img, cv2.ROTATE_180)
        else:
            rotated = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

        score = get_orientation_score(rotated)
        print(f"  Orientation {angle}°: score={score:.1f}")

        if score > best_score:
            best_score = score
            best_angle = angle
            best_img = rotated

    print(f"  Best orientation: {best_angle}° (score={best_score:.1f})")

    # Safe output path regardless of extension
    base, ext = os.path.splitext(image_path)
    out_path = f"{base}_oriented{ext}"
    cv2.imwrite(out_path, best_img)
    return out_path
