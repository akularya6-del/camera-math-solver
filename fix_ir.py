import sympy as sp
from text_to_ir import text_to_ir
from auditor import get_questions_from_html
from solver_engine import MathSolverEngine

qs = get_questions_from_html()
q = qs[13] # Q14 is at index 13
ir = text_to_ir(q["text"], q["options"], q["number"])
solver = MathSolverEngine()
solutions = solver.solve(ir)
print("Solutions:", solutions)
print("Options:", ir.option_ast)
print("Matches:", solver.verify_options(solutions, ir.option_ast))
