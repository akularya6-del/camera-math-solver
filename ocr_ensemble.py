import pytesseract
from PIL import Image
import cv2
import numpy as np

class OCREnsemble:
    def __init__(self):
        try:
            from pix2tex.cli import LatexOCR
            self.latex_ocr = LatexOCR()
        except ImportError:
            self.latex_ocr = None
            print("Warning: pix2tex not installed. Math OCR will fallback to Tesseract.")
            
    def run_text_ocr(self, img_path):
        # We can use pytesseract for now
        img = Image.open(img_path)
        return pytesseract.image_to_string(img, config='--psm 6') # psm 4: Assume a single column of text of variable sizes.
        
    def run_math_ocr(self, img_path):
        if self.latex_ocr is None:
            return ""
        img = Image.open(img_path)
        try:
            return self.latex_ocr(img)
        except Exception as e:
            return ""

    def process_full_image(self, img_path):
        # Very simple version
        text = self.run_text_ocr(img_path)
        return text
