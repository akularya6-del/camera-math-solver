import argparse
import sys
import json
from image_preprocessing import find_best_orientation
from ocr_ensemble import OCREnsemble
from parser import parse_tesseract_to_ir
from solver_engine import MathSolverEngine

def main():
    arg_parser = argparse.ArgumentParser(description="End-to-End Math Solver Pipeline")
    arg_parser.add_argument("image_path", type=str, help="Path to input image")
    arg_parser.add_argument("--no-bank", action="store_true", help="Do not use question bank")
    args = arg_parser.parse_args()
    
    print(f"Processing {args.image_path}...")
    
    # 1. Orientation Detection
    best_img_path = find_best_orientation(args.image_path)
    
    # 2. OCR Ensemble
    ocr = OCREnsemble()
    raw_text = ocr.process_full_image(best_img_path)
    print("--- RAW OCR TEXT ---")
    print(raw_text)
    print("--------------------")
    
    # 3. Parse to Structured IR
    ir = parse_tesseract_to_ir(raw_text)
    print(f"Detected Question {ir.question_number}")
    print(f"Stem AST: {ir.stem_ast}")
    print(f"Option AST: {ir.option_ast}")
    
    # 4. Solver Engine
    solver = MathSolverEngine()
    solutions = solver.solve(ir)
    print(f"Mathematical Solutions: {solutions}")
    
    if solutions:
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
