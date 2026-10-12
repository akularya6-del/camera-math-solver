import argparse
import sys
import json
import os
from image_preprocessing import find_best_orientation
from ocr_ensemble import OCREnsemble
from text_to_ir import text_to_ir
from solver_engine import MathSolverEngine
import re

def parse_full_text(raw_text):
    lines = raw_text.split('\n')
    stem = []
    options = []
    in_question = False
    
    # Simple heuristic to split stem and options
    for i, line in enumerate(lines):
        line_clean = line.strip()
        
        # Starts looking for question content
        if "Q" in line_clean and "." in line_clean:
            in_question = True
            
        if in_question:
            # If line is just a single number or a short formula, it might be an option
            if re.match(r'^[\d\.\-]+$', line_clean) or len(line_clean) < 5:
                options.append(line_clean)
            elif "Mark" not in line_clean and "Clear" not in line_clean and "Save" not in line_clean:
                stem.append(line_clean)
                
    if not stem:
        stem = lines # fallback
        
    # Extract only up to 4 options
    options = [o for o in options if o][-4:]
    
    q_text = " ".join(stem)
    return text_to_ir(q_text, options)

def main():
    arg_parser = argparse.ArgumentParser(description="End-to-End Math Solver Pipeline")
    arg_parser.add_argument("image_path", type=str, help="Path to input image")
    arg_parser.add_argument("--no-bank", action="store_true", help="Do not use question bank")
    args = arg_parser.parse_args()
    
    print(f"Processing {args.image_path}...")
    
    best_img_path = find_best_orientation(args.image_path)
    
    ocr = OCREnsemble()
    raw_text = ocr.process_full_image(best_img_path)
    print("--- RAW OCR TEXT ---")
    print(raw_text)
    print("--------------------")
    
    ir = parse_full_text(raw_text)
    print(f"Stem AST: {ir.stem_ast}")
    print(f"Option AST: {ir.option_ast}")
    print(f"Objective: {ir.objective}")
    
    solver = MathSolverEngine()
    solutions = solver.solve(ir)
    print(f"Mathematical Solutions: {solutions}")
    
    if solutions is not None:
        matched = solver.verify_options(solutions, ir.option_ast)
        print("--- FINAL ANSWER ---")
        if matched:
            for k, v in matched.items():
                print(f"Option {k} ({v})")
        else:
            print("No matching option found.")
    else:
        print("Could not solve algebraically.")

if __name__ == "__main__":
    main()
