import sys
from PIL import Image
from surya.ocr import run_ocr
from surya.model.detection.model import load_model as load_det_model, load_processor as load_det_processor
from surya.model.recognition.model import load_model as load_rec_model
from surya.model.recognition.processor import load_processor as load_rec_processor

print("Loading models...")
det_processor, det_model = load_det_processor(), load_det_model()
rec_model, rec_processor = load_rec_model(), load_rec_processor()

img = Image.open("images/1234.jpeg")

print("Running OCR...")
predictions = run_ocr([img], [[]], det_model, det_processor, rec_model, rec_processor)

for line in predictions[0].text_lines:
    print(line.text)
