import cv2
import numpy as np
from PIL import Image, ImageOps
import pytesseract
from pathlib import Path

def orient_and_crop(image_path: Path):
    im = Image.open(image_path)
    im = ImageOps.exif_transpose(im)
    
    try:
        osd = pytesseract.image_to_osd(im)
        print("OSD Output:")
        print(osd)
    except Exception as e:
        print("OSD failed:", e)

if __name__ == "__main__":
    orient_and_crop(Path("q11_oriented.jpg"))
