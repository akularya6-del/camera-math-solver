import sympy as sp
from models import QuestionIR

class MathSolverEngine:
    def __init__(self):
        # We can add Z3 or other engines later
        pass
        
    def solve(self, ir: QuestionIR):
        # Determine the type of the problem based on natural language or AST
        nl = ir.natural_language.lower()
        
        x = sp.Symbol('x')
        
        if "det" in nl or "determinant" in nl:
            # Determinant equation solver
            # We expect stem_ast to contain an Eq(Determinant(Matrix), value)
            if hasattr(ir.stem_ast, 'lhs') and hasattr(ir.stem_ast, 'rhs'):
                eq = sp.Eq(ir.stem_ast.lhs, ir.stem_ast.rhs)
                solutions = sp.solve(eq, x)
                return solutions
                
        # Fallback to direct solve if stem_ast is just an equation
        if isinstance(ir.stem_ast, sp.Eq):
            return sp.solve(ir.stem_ast)
            
        # If it's a simplification problem
        if ir.stem_ast is not None and not isinstance(ir.stem_ast, sp.Eq):
            return sp.simplify(ir.stem_ast)
            
        return None

    def verify_options(self, solutions, options_ast):
        # options_ast is a dict like {'A': Expr, 'B': Expr, ...}
        results = {}
        for sol in solutions:
            for opt_key, opt_val in options_ast.items():
                if isinstance(sol, sp.Expr) and isinstance(opt_val, sp.Expr):
                    if sp.simplify(sol - opt_val) == 0:
                        results[opt_key] = sol
                        
        return results

