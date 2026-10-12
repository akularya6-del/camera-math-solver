"""
solver_engine.py  —  Universal SymPy-based math solver for CUET MCQs

Handles all objectives produced by text_to_ir.py.
Every path is guarded against None inputs and has a SIGALRM timeout.
"""

import re
import signal
import logging
import sympy as sp
from sympy.solvers.solvers import unrad
from models import QuestionIR

logger = logging.getLogger(__name__)
_SOLVER_TIMEOUT = 15   # seconds per question


# ─────────────────────────── Timeout guard ────────────────────────────────

class _Timeout(Exception):
    pass

def _run_with_timeout(fn, *args, timeout=_SOLVER_TIMEOUT, **kwargs):
    try:
        def _h(s, f): raise _Timeout()
        old = signal.signal(signal.SIGALRM, _h)
        signal.alarm(timeout)
        try:
            return fn(*args, **kwargs)
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old)
    except AttributeError:   # Windows
        return fn(*args, **kwargs)
    except _Timeout:
        raise


# ─────────────────────────── ODE helpers ──────────────────────────────────

def _ode_degree(eq_raw: str):
    eq = eq_raw
    # Substitutions — highest order first
    eq = eq.replace("d²y/dx²", "Y2").replace("d2y/dx2", "Y2")
    eq = eq.replace("d³y/dx³", "Y3").replace("d3y/dx3", "Y3")
    eq = re.sub(r'd\^2y\s*/\s*dx\^2', "Y2", eq)
    eq = re.sub(r'd\^3y\s*/\s*dx\^3', "Y3", eq)
    eq = re.sub(r'\(\s*dy\s*/\s*dx\s*\)', "(Y1)", eq)
    eq = re.sub(r'\bdy\s*/\s*dx\b', "Y1", eq)
    eq = eq.replace("^", "**").replace("[", "(").replace("]", ")")
    eq = eq.replace("²", "**2").replace("³", "**3")
    eq = eq.replace("π", "pi")

    if "=" in eq:
        idx = eq.rfind("=")
        L, R = eq[:idx], eq[idx + 1:]
    else:
        L, R = eq, "0"

    Y1, Y2, Y3 = sp.symbols('Y1 Y2 Y3')
    smap = {'Y1': Y1, 'Y2': Y2, 'Y3': Y3}
    try:
        Ls = sp.sympify(L.strip(), locals=smap)
        Rs = sp.sympify(R.strip(), locals=smap)
    except Exception as e:
        logger.error(f"ODE degree parse failed '{L}' = '{R}': {e}")
        return None

    expr = Ls - Rs
    highest = Y3 if expr.has(Y3) else (Y2 if expr.has(Y2) else Y1)
    try:
        res = unrad(expr)
        if res:
            expr = sp.expand(res[0])
    except Exception:
        expr = sp.expand(expr)
    return [sp.Integer(sp.degree(expr, gen=highest))]


def _ode_order(eq_raw: str):
    if re.search(r'd\^3y|d3y|d³y', eq_raw):
        return [sp.Integer(3)]
    if re.search(r'd\^2y|d2y|d²y', eq_raw):
        return [sp.Integer(2)]
    if re.search(r'dy\s*/\s*dx', eq_raw):
        return [sp.Integer(1)]
    return None


# ─────────────────────────── Main solver class ────────────────────────────

class MathSolverEngine:
    def __init__(self):
        pass

    def solve(self, ir: QuestionIR):
        if ir is None or ir.stem_ast is None:
            return None
        if ir.objective == 'unsupported':
            return None
        try:
            return _run_with_timeout(self._inner, ir, timeout=_SOLVER_TIMEOUT)
        except _Timeout:
            logger.warning(f"Solver timed out [{ir.objective}]")
            return None
        except Exception as e:
            logger.error(f"Solver error [{ir.objective}]: {e}", exc_info=True)
            return None

    def _inner(self, ir: QuestionIR):
        obj = ir.objective
        ast = ir.stem_ast
        x   = sp.Symbol('x', real=True)

        # ── ODE degree ──────────────────────────────────────────────────
        if obj == 'diff_eq_degree':
            return _ode_degree(ast) if isinstance(ast, str) else None

        # ── ODE order ───────────────────────────────────────────────────
        if obj == 'diff_eq_order':
            return _ode_order(ast) if isinstance(ast, str) else None

        # ── Continuity / find param ──────────────────────────────────────
        if obj == 'find_param_continuity':
            if not isinstance(ast, tuple) or len(ast) < 5:
                return None
            func, point, given_val, param, pmap = ast
            px = pmap.get('x', x)
            try:
                lim = sp.limit(func, px, point)
            except Exception as e:
                logger.error(f"limit failed: {e}")
                return None
            sols = sp.solve(sp.Eq(lim, given_val), param)
            return [s for s in sols if s.is_real is not False] or sols

        # ── Pre-image count ──────────────────────────────────────────────
        if obj == 'pre_image_count':
            if not isinstance(ast, list):
                return None
            found = set()
            rx = sp.Symbol('x', real=True)
            for eq in ast:
                try:
                    for s in sp.solve(eq, rx):
                        if s.is_real is not False and s.is_finite is not False:
                            found.add(sp.nsimplify(s))
                except Exception:
                    pass
            return [sp.Integer(len(found))]

        # ── Combinatorics ────────────────────────────────────────────────
        if obj == 'combinatorics':
            if not isinstance(ast, tuple) or len(ast) < 3:
                return None
            nv, rv, is_perm = ast
            if is_perm:
                result = sp.factorial(nv) / sp.factorial(nv - rv)
            else:
                result = sp.binomial(nv, rv)
            return [sp.Integer(result)]

        # ── Definite integral ────────────────────────────────────────────
        if obj == 'definite_integral':
            if not isinstance(ast, tuple) or len(ast) < 4:
                return None
            expr, var, lo, hi = ast
            if expr is None:
                return None
            result = sp.integrate(expr, (var, lo, hi))
            return [sp.simplify(result)]

        # ── Indefinite integral ──────────────────────────────────────────
        if obj == 'integrate':
            if not isinstance(ast, tuple) or len(ast) < 2:
                return None
            expr, var = ast
            if expr is None:
                return None
            return [sp.integrate(expr, var)]

        # ── Differentiation ──────────────────────────────────────────────
        if obj == 'differentiate':
            if not isinstance(ast, tuple) or len(ast) < 2:
                return None
            expr, var = ast[0], ast[1]
            if expr is None:
                return None
            return [sp.simplify(sp.diff(expr, var))]

        # ── Derivative at a point ─────────────────────────────────────────
        if obj == 'differentiate_at':
            if not isinstance(ast, tuple) or len(ast) < 3:
                return None
            expr, var, pt = ast
            if expr is None:
                return None
            deriv = sp.diff(expr, var)
            return [sp.simplify(deriv.subs(var, pt))]

        # ── Maxima / minima ──────────────────────────────────────────────
        if obj == 'extremum':
            if not isinstance(ast, tuple) or len(ast) < 3:
                return None
            expr, var, mode = ast
            if expr is None:
                return None
            deriv1 = sp.diff(expr, var)
            crit = sp.solve(deriv1, var)
            if not crit:
                return None
            vals = [expr.subs(var, cp) for cp in crit if sp.im(cp) == 0]
            if not vals:
                return None
            result = max(vals) if mode == 'max' else min(vals)
            return [sp.simplify(result)]

        # ── Tangent slope ────────────────────────────────────────────────
        if obj == 'tangent_slope':
            if not isinstance(ast, tuple) or len(ast) < 3:
                return None
            expr, var, pt = ast
            if expr is None:
                return None
            slope = sp.diff(expr, var).subs(var, pt)
            return [sp.simplify(slope)]

        # ── Matrix operation ─────────────────────────────────────────────
        if obj == 'matrix_op':
            if not isinstance(ast, tuple) or len(ast) < 2:
                return None
            op, mat = ast
            if mat is None:
                return None
            if op == 'determinant':
                return [mat.det()]
            if op == 'trace':
                return [mat.trace()]
            if op == 'inverse':
                try:
                    return [mat.inv()]
                except Exception:
                    return None
            if op == 'transpose':
                return [mat.T]
            return [mat.det()]   # default

        # ── Solve equation ───────────────────────────────────────────────
        if obj == 'solve_eq':
            if not isinstance(ast, sp.Eq):
                return None
            # Detect the unknown (prefer x, then k, then any free symbol)
            unknowns = sorted(ast.free_symbols, key=lambda s: str(s))
            target = None
            for pref in ('x', 'k', 'n', 'y'):
                cands = [s for s in unknowns if str(s) == pref]
                if cands:
                    target = cands[0]
                    break
            if target is None and unknowns:
                target = unknowns[0]
            if target is None:
                return None
            sols = sp.solve(ast, target)
            return [s for s in sols if s.is_real is not False] or sols

        # ── Inverse trig ─────────────────────────────────────────────────
        if obj == 'inverse_trig':
            if not isinstance(ast, tuple) or not ast:
                return None
            expr = ast[0]
            if expr is None:
                return None
            val = sp.simplify(expr)
            return [val]

        # ── Probability ──────────────────────────────────────────────────
        if obj == 'probability':
            if not isinstance(ast, tuple) or len(ast) < 3:
                return None
            pa, pb, mode, indep = ast
            if mode == 'union':
                if indep:
                    result = pa + pb - pa * pb
                else:
                    result = pa + pb   # lower bound without P(A∩B)
            else:
                result = pa * pb if indep else sp.Symbol('P_AB')
            return [sp.simplify(result)]

        # ── Vector identity ───────────────────────────────────────────────
        if obj == 'vector_identity':
            return [ast]

        # ── Evaluate / simplify expression ────────────────────────────────
        if obj == 'evaluate_expr':
            if not isinstance(ast, tuple) or not ast:
                return None
            expr = ast[0]
            if expr is None:
                return None
            val = sp.simplify(expr)
            try:
                # Try numeric evaluation
                n_val = complex(val)
                if n_val.imag == 0:
                    return [sp.nsimplify(n_val.real)]
            except Exception:
                pass
            return [val]

        return None

    # ─────────────────────── verify_options ───────────────────────────────

    def verify_options(self, computed_solutions, options_ast: dict):
        """Match computed solutions against MCQ options symbolically."""
        if not computed_solutions or not options_ast:
            return None

        matched = {}
        for sol in computed_solutions:
            for letter, opt in options_ast.items():
                if isinstance(opt, str):
                    if str(sol).strip() == opt.strip():
                        matched[letter] = opt
                    continue
                if isinstance(sol, str):
                    continue
                try:
                    if sol == opt:
                        matched[letter] = opt
                        continue
                    if hasattr(sol, 'equals') and sol.equals(opt):
                        matched[letter] = opt
                        continue
                    diff = sp.simplify(sol - opt)
                    if diff == 0:
                        matched[letter] = opt
                        continue
                    if diff.is_number:
                        try:
                            if abs(float(diff)) < 1e-9:
                                matched[letter] = opt
                                continue
                        except Exception:
                            pass
                except Exception:
                    pass
        return matched if matched else None
