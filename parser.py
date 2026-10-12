import re
import sympy as sp
from models import QuestionIR, OCRToken
from text_to_ir import text_to_ir

def parse_tesseract_to_ir(raw_text: str) -> QuestionIR:
    lines = raw_text.split("\n")
    stem = []
    options = []
    
    current_mode = "stem"
    q_number = None
    
    for line in lines:
        line = line.strip()
        if not line: continue
        
        q_match = re.search(r'Q(\d+)', line)
        if q_match and not q_number:
            q_number = int(q_match.group(1))
            
        if re.match(r'^[A-D][\.\)]\s*', line):
            current_mode = "options"
            opt_text = line[2:].strip()
            options.append(opt_text)
        else:
            if current_mode == "stem":
                stem.append(line)
                
    nl = " ".join(stem)
    
    # Delegate to the powerful text_to_ir parser we wrote!
    return text_to_ir(nl, options, q_number)
