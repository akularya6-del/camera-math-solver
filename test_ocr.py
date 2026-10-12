import cv2
from PIL import Image
import pytesseract
import sys

def test_ocr(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Resize down if it's too huge, Tesseract struggles with 4K resolution
    h, w = gray.shape
    if w > 1920:
        ratio = 1920.0 / w
        gray = cv2.resize(gray, (int(w*ratio), int(h*ratio)))
        
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 10)
    
    cv2.imwrite("test_thresh.jpg", thresh)
    
    pil_img = Image.fromarray(thresh)
    text = pytesseract.image_to_string(pil_img, config='--psm 6')
    print("PSM 6:", text)

test_ocr("images/1234.jpeg")
