from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float

@dataclass
class OCRToken:
    text: str
    confidence: float
    box: BoundingBox
    alternatives: List[str] = field(default_factory=list)

@dataclass
class QuestionIR:
    question_number: Optional[int]
    natural_language: str
    math_expressions: List[str]
    stem_ast: Any
    option_ast: Dict[str, Any]
    raw_ocr_candidates: List[OCRToken]
    bounding_boxes: Dict[str, BoundingBox]
    uncertainty_tokens: List[OCRToken]
    parse_confidence: float
