from PIL import Image
import os
from pix2tex.cli import LatexOCR

def main():
    print("Loading LatexOCR...")
    model = LatexOCR()
    print("Processing images/1234.jpeg...")
    img = Image.open("images/1234.jpeg")
    print("Result:", model(img))
    
    print("Processing images/12.jpeg...")
    img = Image.open("images/12.jpeg")
    print("Result:", model(img))

    print("Processing images/123.jpeg...")
    img = Image.open("images/123.jpeg")
    print("Result:", model(img))
    
main()
