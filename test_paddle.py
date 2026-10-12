from paddleocr import PaddleOCR
import sys

ocr = PaddleOCR(use_angle_cls=True, lang='en')
img_path = "images/1234.jpeg"
result = ocr.ocr(img_path, cls=True)
for idx in range(len(result)):
    res = result[idx]
    for line in res:
        print(line[1][0])
