import cv2
import pytesseract
import numpy as np
from PIL import Image

def get_orientation_score(image_cv):
    # Use Tesseract to get some basic text data
    # psm 6: Assume a single uniform block of text.
    data = pytesseract.image_to_data(image_cv, config='--psm 6', output_type=pytesseract.Output.DICT)
    
    # Calculate score based on recognized text and confidence
    valid_text_count = 0
    total_conf = 0
    
    for i, text in enumerate(data['text']):
        text = text.strip()
        conf = float(data['conf'][i])
        if len(text) > 2 and conf > 40:  # arbitrary threshold for "valid" token
            valid_text_count += 1
            total_conf += conf
            
    # Number of English/math tokens as a simple metric
    return valid_text_count * total_conf

def find_best_orientation(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read {image_path}")
        
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
        elif angle == 270:
            rotated = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            
        score = get_orientation_score(rotated)
        print(f"Angle {angle}: Score {score}")
        
        if score > best_score:
            best_score = score
            best_angle = angle
            best_img = rotated
            
    print(f"Best orientation is {best_angle} degrees with score {best_score}")
    
    out_path = image_path.replace(".jpg", f"_best_{best_angle}.jpg")
    cv2.imwrite(out_path, best_img)
    return out_path

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        find_best_orientation(sys.argv[1])
    else:
        find_best_orientation("q11_oriented.jpg")
