import re
import json

def extract_questions():
    with open("exam_source.html", "r", encoding="utf-8") as f:
        html = f.read()

    # Find the const questions = [ ... ]; block
    match = re.search(r'const\s+questions\s*=\s*\[(.*?)\];', html, re.DOTALL)
    if not match:
        raise ValueError("Could not find questions block")
    
    content = match.group(1)
    
    # We will use regex to find { text: "...", options: [...], answer: X }
    pattern = re.compile(
        r'\{\s*text:\s*"(.*?)",\s*options:\s*\[(.*?)\],\s*answer:\s*(\d+)\s*\}',
        re.DOTALL
    )
    
    questions = []
    for i, m in enumerate(pattern.finditer(content)):
        text = m.group(1)
        options_raw = m.group(2)
        answer_idx = int(m.group(3))
        
        # split options by string
        opts = []
        for opt_match in re.finditer(r'"(.*?)"', options_raw):
            opts.append(opt_match.group(1))
            
        questions.append({
            "number": i + 1,
            "text_html": text,
            "options_html": opts,
            "answer_idx": answer_idx
        })
        
    return questions

if __name__ == "__main__":
    q = extract_questions()
    with open("question_bank_with_answers.json", "w", encoding="utf-8") as f:
        json.dump(q, f, indent=2, ensure_ascii=False)
    print(f"Extracted {len(q)} questions.")
