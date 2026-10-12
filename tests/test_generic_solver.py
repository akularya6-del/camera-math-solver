import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sympy as sp
from models import QuestionIR
from solver_engine import MathSolverEngine

def test_solve_eq():
    x = sp.Symbol('x')
    ast = sp.Eq(10 - 3*x, 4)
    ir = QuestionIR(question_number=11, natural_language="", math_expressions=[], stem_ast=ast, option_ast={}, objective="solve_eq")
    solver = MathSolverEngine()
    assert solver.solve(ir) == [2]

def test_differentiate():
    x = sp.Symbol('x')
    ast = (sp.exp(x * sp.sin(x)), x)
    ir = QuestionIR(question_number=1, natural_language="", math_expressions=[], stem_ast=ast, option_ast={}, objective="differentiate")
    solver = MathSolverEngine()
    sol = solver.solve(ir)[0]
    expected = (x*sp.cos(x) + sp.sin(x))*sp.exp(x*sp.sin(x))
    assert sp.simplify(sol - expected) == 0

def test_tangent_slope():
    x = sp.Symbol('x')
    ast = (x**3 - x, x, 2)
    ir = QuestionIR(question_number=1, natural_language="", math_expressions=[], stem_ast=ast, option_ast={}, objective="tangent_slope")
    solver = MathSolverEngine()
    assert solver.solve(ir) == [11]

def test_matrix_op():
    A = sp.Matrix([[1, 2], [3, 4]])
    ast = A + A.T
    ir = QuestionIR(question_number=1, natural_language="", math_expressions=[], stem_ast=ast, option_ast={}, objective="matrix_op")
    solver = MathSolverEngine()
    expected = sp.Matrix([[2, 5], [5, 8]])
    assert solver.solve(ir) == [expected]
    
def test_integrate():
    x = sp.Symbol('x')
    ast = (x * sp.exp(x), x)
    ir = QuestionIR(question_number=1, natural_language="", math_expressions=[], stem_ast=ast, option_ast={}, objective="integrate")
    solver = MathSolverEngine()
    sol = solver.solve(ir)[0]
    expected = (x - 1)*sp.exp(x)
    assert sp.simplify(sol - expected) == 0

if __name__ == "__main__":
    test_solve_eq()
    test_differentiate()
    test_tangent_slope()
    test_matrix_op()
    test_integrate()
    print("All generic solver tests passed!")
