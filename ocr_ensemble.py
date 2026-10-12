"""
ocr_ensemble.py
PaddleOCR wrapper with robust result parsing.

Failure modes fixed:
  1. cv2.imread returns None if path is bad → crash on shape[:2] → guard added
  2. PaddleOCR init crash is silently swallowed but self.paddle stays None →
     proper error logged so operator can diagnose
  3. Only 2000px cap → very small images get upscaled (bad) → min cap too
  4. result[0] may be None even if result is a list → crash → guard added
  5. rec_scores not filtered → low-confidence garbage text leaks into OCR output
  6. result[0]['rec_texts'] joined without score filtering → added threshold
"""

import cv2
import warnings
import logging

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

OCR_SCORE_THRESHOLD = 0.5   # discard Paddle lines with confidence < this


class OCREnsemble:
    def __init__(self):
        self.paddle = None
        try:
            from paddleocr import PaddleOCR
            self.paddle = PaddleOCR(lang='en')
            logger.info("PaddleOCR loaded successfully.")
        except ImportError:
            logger.error("PaddleOCR not installed.  Run: pip install paddleocr paddlepaddle")
        except Exception as e:
            logger.error(f"PaddleOCR failed to init: {e}")

    def run_text_ocr(self, img_path: str) -> str:
        if not self.paddle:
            return ""

        img = cv2.imread(img_path)
        if img is None:
            logger.error(f"Cannot read image at path: {img_path}")
            return ""

        h, w = img.shape[:2]
        # Clamp to [800, 2000] px on the long side for best OCR accuracy
        max_side = max(h, w)
        if max_side > 2000:
            ratio = 2000.0 / max_side
            img = cv2.resize(img, (int(w * ratio), int(h * ratio)))
        elif max_side < 800:
            ratio = 800.0 / max_side
            img = cv2.resize(img, (int(w * ratio), int(h * ratio)))

        result = None
        try:
            result = self.paddle.ocr(img, cls=False)
        except TypeError:
            try:
                result = self.paddle.ocr(img)
            except Exception as e:
                logger.error(f"PaddleOCR inference error: {e}")
                return ""
        except Exception as e:
            logger.error(f"PaddleOCR inference error: {e}")
            return ""

        return self._extract_text(result)

    def _extract_text(self, result) -> str:
        """Parse PaddleOCR result regardless of API version."""
        if not result:
            return ""

        text_lines = []

        # ── PaddleOCR V3 pipeline format: list of dicts ──────────────────────
        if isinstance(result, list) and len(result) > 0 and isinstance(result[0], dict):
            rec_texts  = result[0].get('rec_texts', [])
            rec_scores = result[0].get('rec_scores', [])

            for i, text in enumerate(rec_texts):
                if not text or not text.strip():
                    continue
                score = rec_scores[i] if i < len(rec_scores) else 1.0
                if score >= OCR_SCORE_THRESHOLD:
                    text_lines.append(text.strip())

            if text_lines:
                return "\n".join(text_lines)

        # ── PaddleOCR V2 format: list[list[line]] ────────────────────────────
        if isinstance(result, list):
            page = result[0] if result else []
            if page is None:
                return ""
            for line in page:
                try:
                    text  = line[1][0]
                    score = line[1][1]
                    if text and text.strip() and score >= OCR_SCORE_THRESHOLD:
                        text_lines.append(text.strip())
                except (IndexError, TypeError):
                    pass

        return "\n".join(text_lines)

    def process_full_image(self, img_path: str) -> str:
        return self.run_text_ocr(img_path)
