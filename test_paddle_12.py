from paddleocr import PaddleOCR
ocr = PaddleOCR(lang='en')
result = ocr.ocr("images/12.jpeg")
print(result)
