from paddleocr import PaddleOCR
from PIL import Image

class OCREnsemble:
    def __init__(self):
        # We initialize PaddleOCR here
        self.p_ocr = PaddleOCR(use_angle_cls=True, lang='en')
        
    def extract_text(self, image_path: str):
        # Using predict as required by newer version
        result = self.p_ocr.predict(image_path)
        texts = []
        if hasattr(result, 'get_res'): # Handle new pipeline API
            for r in result:
                for line in r['rec_text']:
                    texts.append(line)
        return texts
