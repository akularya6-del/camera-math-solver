import re
import json
import sympy as sp
import logging
from solver_engine import MathSolverEngine
from text_to_ir import text_to_ir

logging.basicConfig(level=logging.INFO)

def get_questions_from_html(file_path="exam_source.html"):
    with open(file_path, "r", encoding="utf-8") as f:
        html = f.read()
    match = re.search(r'const\s+questions\s*=\s*\[(.*?)\];', html, re.DOTALL)
    if not match: raise ValueError("Could not find questions array")
    content = match.group(1)
    parts = content.split("},")
    questions = []
    
    for i, part in enumerate(parts):
        part = part.strip()
        if not part: continue
        texts = re.findall(r'text:\s*"(.*?)"', part)
        if not texts: continue
        text = texts[-1]
        
        options_matches = re.findall(r'options:\s*\[(.*?)\]', part)
        if options_matches:
            opt_raw = options_matches[-1]
            options = [m.group(1) for m in re.finditer(r'"(.*?)"', opt_raw)]
        else:
            options = []
            
        answer_matches = re.findall(r'answer:\s*(\d+)', part)
        answer = int(answer_matches[-1]) if answer_matches else -1
        
        questions.append({
            "number": i+1,
            "text": text,
            "options": options,
            "source_ans_idx": answer
        })
    return questions

def audit_question(q, solver):
    ans = q["source_ans_idx"]
    
    # 1. Independently solve
    try:
        ir = text_to_ir(q["text"], q["options"], q["number"])
        correct_vals = solver.solve(ir)
    except NotImplementedError:
        return "UNSUPPORTED", -1, []
        
    if correct_vals is None or len(correct_vals) == 0:
        return "AMBIGUOUS_WORDING", -1, []
        
    opts = list(ir.option_ast.values())
        
    # 2. Detect duplicate equivalent options
    equivalent_groups = {}
    for i, o in enumerate(opts):
        for j in range(i+1, len(opts)):
            try:
                if isinstance(o, sp.Expr) and isinstance(opts[j], sp.Expr):
                    if sp.simplify(o - opts[j]) == 0:
                        equivalent_groups.setdefault(i, []).append(j)
            except:
                pass
                    
    # Check which options match the correct values
    matching_indices = []
    for val in correct_vals:
        for idx, o in enumerate(opts):
            try:
                if isinstance(o, sp.Expr) and isinstance(val, sp.Expr):
                    if sp.simplify(o - val) == 0:
                        matching_indices.append(idx)
                elif isinstance(o, sp.Matrix) and isinstance(val, sp.Matrix):
                    if o == val:
                        matching_indices.append(idx)
                elif o == val:
                    matching_indices.append(idx)
            except:
                pass
                
    matching_indices = list(set(matching_indices))
    
    if len(matching_indices) == 0:
        return "NO_VALID_OPTION", -1, []
    elif len(matching_indices) > 1:
        return "MULTIPLE_VALID_OPTIONS", matching_indices[0], matching_indices
        
    correct_idx = matching_indices[0]
    if correct_idx == ans:
        return "VALID", correct_idx, matching_indices
    else:
        return "BAD_KEY", correct_idx, matching_indices


def main():
    questions = get_questions_from_html()
    results = []
    solver = MathSolverEngine()
    
    for q in questions:
        status, correct_idx, all_matches = audit_question(q, solver)
        if status != "UNSUPPORTED":
            results.append({
                "number": q["number"],
                "text": q["text"],
                "status": status,
                "source_ans": q["source_ans_idx"],
                "math_ans": correct_idx,
                "all_matches": all_matches
            })
            
    with open("corrections.json", "w") as f:
        json.dump(results, f, indent=2)
        
    logging.info(f"Audited {len(questions)} questions. Generated {len(results)} audit records.")
    for r in results:
        print(f"Q{r['number']}: {r['status']} | Source: {r['source_ans']}, Computed: {r['math_ans']}")

if __name__ == "__main__":
    main()
