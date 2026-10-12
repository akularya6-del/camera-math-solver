import cv2
import numpy as np
import pytesseract
from PIL import Image

def process_image(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. Enhance contrast (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # 2. Adaptive Thresholding
    thresh = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 15)
    
    # 3. Dilation and Erosion to remove noise (closing)
    kernel = np.ones((2,2), np.uint8)
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    cv2.imwrite("debug_thresh.jpg", opened)
    
    # 4. OCR
    text = pytesseract.image_to_string(opened, config='--psm 6')
    print("Extracted:")
    print(text)

process_image("images/1234.jpeg")
