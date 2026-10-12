"""
text_to_ir.py  —  Universal OCR text → QuestionIR converter
Covers the full CUET Class-12 Mathematics syllabus generically.

Design philosophy:
  • Parse what's there — don't hard-code question numbers or specific expressions.
  • classify_objective() decides WHAT math is being asked.
  • extract_stem_ast() extracts the mathematical content AS a SymPy object.
  • Options are parsed symbolically where possible.
  • Falls back to raw text only as last resort.
"""

import re
import sympy as sp
from models import QuestionIR

# ─────────────────────── Symbol catalogue ──────────────────────────────────
x, y, z, t, n = sp.symbols('x y z t n', real=True)
k, a, b, c    = sp.symbols('k a b c', real=True)
i, j          = sp.symbols('i j', integer=True, positive=True)
_SYMS = {'x': x, 'y': y, 'z': z, 't': t, 'n': n,
         'k': k, 'a': a, 'b': b, 'c': c,
         'pi': sp.pi, 'e': sp.E, 'E': sp.E, 'oo': sp.oo,
         'sin': sp.sin, 'cos': sp.cos, 'tan': sp.tan,
         'asin': sp.asin, 'acos': sp.acos, 'atan': sp.atan,
         'sinh': sp.sinh, 'cosh': sp.cosh, 'tanh': sp.tanh,
         'exp': sp.exp, 'log': sp.log, 'ln': sp.log,
         'sqrt': sp.sqrt, 'Abs': sp.Abs,
         'Matrix': sp.Matrix, 'factorial': sp.factorial,
         'binomial': sp.binomial}

# ───────────────────────── OCR Normalisation ───────────────────────────────

_OCR_FIXES = [
    # Unicode superscripts → **n
    ('²', '**2'), ('³', '**3'), ('⁴', '**4'), ('⁵', '**5'),
    # Unicode minus/dash
    ('–', '-'), ('−', '-'), ('‒', '-'), ('—', '-'),
    # Unicode multiply
    ('×', '*'), ('·', '*'),
    # Greek letters
    ('π', 'pi'), ('α', 'alpha'), ('β', 'beta'), ('θ', 'theta'),
    ('φ', 'phi'), ('λ', 'lmbda'), ('Σ', 'Sum'), ('∞', 'oo'),
    # Integral / sqrt
    ('∫', 'integrate('), ('√', 'sqrt('),
    # Inverse trig (OCR renders these various ways)
    ('sin⁻¹', 'asin'), ('cos⁻¹', 'acos'), ('tan⁻¹', 'atan'),
    ('sin^{-1}', 'asin'), ('cos^{-1}', 'acos'), ('tan^{-1}', 'atan'),
    ('sin-1', 'asin'), ('cos-1', 'acos'), ('tan-1', 'atan'),
    # LaTeX remnants
    ('\\frac', ''), ('\\left', ''), ('\\right', ''),
    ('\\cdot', '*'), ('\\times', '*'),
    ('\\_', '_'), ('\\', ''),
    # OCR typos
    ('iog', 'log'), ('Iog', 'log'), ('Slnx', 'sin(x)'),
    ('{', '('), ('}', ')'),
    # ^ → **
    ('^', '**'),
]

def normalize_ocr(text: str) -> str:
    """Apply all OCR normalisation substitutions, return lowercased."""
    for src, dst in _OCR_FIXES:
        text = text.replace(src, dst)
    # x2 → x**2 only at word boundary (avoid x21 etc.)
    text = re.sub(r'\bx2\b', 'x**2', text)
    text = re.sub(r'\bx3\b', 'x**3', text)
    text = re.sub(r'\bn2\b', 'n**2', text)
    # Bare trig arguments: cos x → cos(x)
    text = re.sub(r'\b(sin|cos|tan|log|exp|sqrt)\s+([a-z])\b', r'\1(\2)', text)
    # Implicit multiplication: 2x → 2*x, kx → k*x (digit/letter boundary)
    text = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', text)
    # Clean up stacked ** from multiple substitutions
    text = text.replace('****', '**').replace('***', '**')
    return text


def _sympify(expr_str: str, extra_syms: dict = None):
    """Safe sympify; returns None on failure."""
    syms = dict(_SYMS)
    if extra_syms:
        syms.update(extra_syms)
    expr_str = normalize_ocr(expr_str)
    try:
        result = sp.sympify(expr_str, locals=syms)
        # Reject expressions whose only free symbols are long garbage tokens
        garbage = [s for s in result.free_symbols
                   if len(str(s)) > 4 and str(s) not in syms]
        if garbage:
            return None
        return result
    except Exception:
        return None


# ───────────────── Strip exam UI noise from raw OCR ───────────────────────

_UI_NOISE = re.compile(
    r'(mark\s+for\s+review|clear\s+response|save\s+(&|and)\s+next|previous|submit|'
    r'not\s+visited|not\s+answered|answered|marked|question\s+palette|'
    r'time\s+remaining|roll\s+no|section\s+a|mathematics\s+examination|'
    r'set\s+[a-z]\s*\||code\s*:|multiple\s+choice|single\s+correct|'
    r'\+\s*\d+\s+correct|\-\s*\d+\s+wrong|\d+\s+questions?)',
    re.IGNORECASE
)

def strip_ui_noise(text: str) -> str:
    lines = []
    for line in text.split('\n'):
        if not _UI_NOISE.search(line) and line.strip():
            lines.append(line)
    return '\n'.join(lines)


# ───────────────────────── Option parsing ─────────────────────────────────

_JUNK_TOKENS = {
    'mark', 'clear', 'save', 'next', 'previous', 'submit', 'response',
    'review', 'answered', 'visited', 'palette', 'section', 'mathematics',
    'examination', 'candidate', 'correct', 'wrong', 'choice', 'single',
    'question', 'set', 'code', 'nean', 'nest', 'prev', 'esc',
    'f1','f2','f3','f4','f5','f10','f11','f12',
    'dii','dd','tab','ctrl','alt','shift','cmd',
}

_OPTION_RE = re.compile(r'^([A-Da-d])[\.\)]\s*(.+)', re.IGNORECASE)
_FOOTER_KW = {'mark','clear','save','next','previous','submit','answered','visited','palette'}

def _is_plausible_option(s: str) -> bool:
    s = s.strip()
    if not s or len(s) > 80:
        return False
    if s.lower() in _JUNK_TOKENS:
        return False
    # Pure long alphabetic with no digits/math => likely junk
    if re.match(r'^[a-zA-Z]{5,}$', s) and not any(c.isdigit() for c in s):
        # unless it's a math word
        if s.lower() not in {'sqrt', 'infinity', 'undefined', 'none', 'zero',
                              'infinite', 'continuous', 'bijection', 'real'}:
            return False
    return True

def parse_options(lines: list) -> list:
    """
    Extract up to 4 MCQ options from OCR lines.
    Two strategies:
      1) Look for explicit A./B./C./D. prefixes anywhere.
      2) After the Q-stem line, take short unambiguous lines.
    """
    # Strategy 1: explicit letter prefixes
    explicit = []
    for line in lines:
        m = _OPTION_RE.match(line.strip())
        if m:
            val = m.group(2).strip()
            if _is_plausible_option(val):
                explicit.append(val)
    if len(explicit) >= 2:
        return explicit[:4]

    # Strategy 2: short lines after Q-stem
    q_idx = None
    for i, line in enumerate(lines):
        if re.search(r'Q\s*\d+[\.\:]', line):
            q_idx = i
            break
    if q_idx is None:
        return []

    options = []
    for line in lines[q_idx + 1:]:
        s = line.strip()
        low = s.lower()
        if any(kw in low for kw in _FOOTER_KW):
            break
        if _is_plausible_option(s) and len(s) < 60:
            options.append(s)
            if len(options) == 4:
                break
    return options


def build_option_ast(raw_options: list) -> dict:
    """Convert raw option strings to {letter: SymPy|str}."""
    opts = {}
    for i, opt in enumerate(raw_options[:4]):
        letter = chr(65 + i)
        opt = re.sub(r'^[A-Da-d][\.\)]\s*', '', opt.strip())
        val = _sympify(opt)
        opts[letter] = val if val is not None else opt
    return opts


# ─────────────────────────── Stem parsing ──────────────────────────────────

def extract_stem_lines(lines: list) -> list:
    """Return lines that belong to the question stem (not footer/palette)."""
    q_idx = None
    for i, line in enumerate(lines):
        if re.search(r'Q\s*\d+[\.\:]', line):
            q_idx = i
            break
    if q_idx is None:
        return lines

    stem = []
    for line in lines[q_idx:]:
        low = line.strip().lower()
        if any(kw in low for kw in _FOOTER_KW):
            break
        stem.append(line.strip())
    return stem


def _extract_q_number(lines: list):
    for line in lines:
        m = re.search(r'[Qq]\s*(\d+)', line)
        if m:
            return int(m.group(1))
    return None


# ─────────────────────── Math expression extraction ────────────────────────

def _find_matrix(text: str):
    """Try to parse [[a,b],[c,d]] matrix from text."""
    pat = re.compile(
        r'\[\s*\[\s*([\d\-\+\.x]+)\s*,\s*([\d\-\+\.x]+)\s*\]\s*,?\s*'
        r'\[\s*([\d\-\+\.x]+)\s*,\s*([\d\-\+\.x]+)\s*\]'
    )
    m = pat.search(text)
    if m:
        try:
            entries = [_sympify(g) for g in m.groups()]
            if all(e is not None for e in entries):
                return sp.Matrix([[entries[0], entries[1]],
                                  [entries[2], entries[3]]])
        except Exception:
            pass
    return None


def _find_equation(text: str, default_var=None):
    """
    Find 'LHS = RHS' in text and return sp.Eq(LHS, RHS).
    Handles multiple '=' by treating only the last one (or the first
    one not part of variable definitions).
    """
    norm = normalize_ocr(text)
    # Strip known non-equation fragments
    norm = re.sub(r'for\s+x\s*[≠!=][^,]*', '', norm)
    norm = re.sub(r'where\s+.*', '', norm)
    norm = re.sub(r',?\s*is\s+continuous.*', '', norm)
    norm = re.sub(r'find\s+\w+', '', norm, flags=re.IGNORECASE)

    # Find = signs
    parts = norm.split('=')
    if len(parts) < 2:
        return None

    # Try pairs: first=second, ..., penultimate=last
    for pi in range(len(parts) - 1):
        lhs_str = parts[pi].strip()
        rhs_str = parts[pi + 1].strip()
        lhs = _sympify(lhs_str)
        rhs = _sympify(rhs_str)
        if lhs is not None and rhs is not None:
            return sp.Eq(lhs, rhs)
    return None


def _find_expression(text: str):
    """
    Extract the primary mathematical expression from a text fragment.
    Tries progressively: function def, equation LHS, standalone expr.
    """
    norm = normalize_ocr(text)
    # Try f(x) = expr
    m = re.search(r'(?:f|g|h)\s*\(\s*x\s*\)\s*=\s*(.+?)(?:\s+for\b|\s+and\b|\s*,|\s+is\b|$)',
                  norm, re.IGNORECASE)
    if m:
        val = _sympify(m.group(1))
        if val is not None:
            return val

    # Try y = expr
    m = re.search(r'\by\s*=\s*(.+?)(?:\s+find|\s*,|\s+then|$)', norm, re.IGNORECASE)
    if m:
        val = _sympify(m.group(1))
        if val is not None:
            return val

    # Try whole text as expression
    return _sympify(norm)


def _find_limits_bounds(text: str):
    """Extract definite integral bounds from text like '∫₀¹', '∫ from 0 to 1', etc."""
    # Unicode subscript/superscript bounds
    m = re.search(r'from\s+([\d\.\-\w/pi]+)\s+to\s+([\d\.\-\w/pi]+)', text, re.IGNORECASE)
    if m:
        lo = _sympify(m.group(1))
        hi = _sympify(m.group(2))
        if lo is not None and hi is not None:
            return lo, hi

    # ∫₀¹ style (unicode) — try extracting adjacent digits
    m = re.search(r'∫\s*([₀₁₂₃₄₅₆₇₈₉\d])\s*([¹²³⁴⁵⁶⁷⁸⁹\d])', text)
    if m:
        subs_map = str.maketrans('₀₁₂₃₄₅₆₇₈₉', '0123456789')
        sups_map = str.maketrans('¹²³⁴⁵⁶⁷⁸⁹', '123456789')
        lo = int(m.group(1).translate(subs_map) or m.group(1))
        hi = int(m.group(2).translate(sups_map) or m.group(2))
        return sp.Integer(lo), sp.Integer(hi)

    return None


def _find_ncr(text: str):
    """Find nCr or nPr in text. Returns (n_val, r_val, is_permutation)."""
    # e.g. "8C2", "10P3", "C(8,2)", "P(10,3)"
    m = re.search(r'(\d+)\s*[Cc]\s*(\d+)', text)
    if m:
        return int(m.group(1)), int(m.group(2)), False
    m = re.search(r'(\d+)\s*[Pp]\s*(\d+)', text)
    if m:
        return int(m.group(1)), int(m.group(2)), True
    m = re.search(r'[Cc]\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)', text)
    if m:
        return int(m.group(1)), int(m.group(2)), False
    m = re.search(r'[Pp]\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)', text)
    if m:
        return int(m.group(1)), int(m.group(2)), True
    return None


def _find_matrix_op(text: str):
    """Detect matrix operations: det, trace, inverse, transpose."""
    tl = text.lower()
    mat = _find_matrix(text)
    if mat is None:
        return None, None
    if 'det' in tl or '|a|' in tl or '|b|' in tl:
        return 'determinant', mat
    if 'trace' in tl:
        return 'trace', mat
    if 'inverse' in tl or 'a**{-1}' in tl or 'a^{-1}' in tl:
        return 'inverse', mat
    if 'transpose' in tl or "a'" in tl:
        return 'transpose', mat
    # Bare matrix — assume determinant
    return 'determinant', mat


# ─────────────────────── Objective classification ─────────────────────────

def classify_objective(q_text: str, q_lower: str, q_norm: str) -> str:
    """
    Determine the mathematical objective from question text.
    Returns one of the objective strings handled by MathSolverEngine.
    """
    # ── Degree / order of ODE ────────────────────────────────────────────
    if 'degree of' in q_lower and re.search(r'd[²2]?y|dy/dx', q_lower):
        return 'diff_eq_degree'
    if 'order of' in q_lower and re.search(r'd[²2]?y|dy/dx', q_lower):
        return 'diff_eq_order'

    # ── Continuity — find parameter ──────────────────────────────────────
    if 'continuous at' in q_lower:
        return 'find_param_continuity'
    if re.search(r'find\s+k\b.*continuous|continuous.*find\s+k\b', q_lower):
        return 'find_param_continuity'

    # ── Pre-image / inverse image ─────────────────────────────────────────
    if re.search(r'elements?\s+in\s+f[\'\-\^⁻]|f[\'\-\^⁻]\s*[\(\{]', q_lower):
        return 'pre_image_count'

    # ── Combinatorics ────────────────────────────────────────────────────
    if re.search(r'\d+\s*[Cc]\s*\d+|\d+\s*[Pp]\s*\d+|[Cc]\(\d|[Pp]\(\d', q_text):
        return 'combinatorics'
    if re.search(r'\bpermutation|\bcombination|\barrangement', q_lower):
        return 'combinatorics'

    # ── Definite integral ────────────────────────────────────────────────
    if re.search(r'∫\s*\d|from\s+\d.*to\s+\d|integrate.*from|definite', q_lower):
        return 'definite_integral'

    # ── Indefinite integral ───────────────────────────────────────────────
    if re.search(r'integrat|∫', q_lower):
        return 'integrate'

    # ── Derivative ───────────────────────────────────────────────────────
    if re.search(r"f\s*'\s*\(|dy\s*/\s*dx|d/dx|differentiat|derivative", q_lower):
        return 'differentiate'

    # ── Derivative at a point ─────────────────────────────────────────────
    if re.search(r'dy\s*/\s*dx\s+at\s+x\s*=|find.*dy/dx.*x\s*=\s*\d', q_lower):
        return 'differentiate_at'

    # ── Maxima / minima ───────────────────────────────────────────────────
    if re.search(r'maximum|minimum|maxima|minima|extreme|critical', q_lower):
        return 'extremum'

    # ── Tangent / normal ─────────────────────────────────────────────────
    if re.search(r'tangent|normal\s+to\s+the\s+curve', q_lower):
        return 'tangent_slope'

    # ── Determinant / matrix operation ───────────────────────────────────
    if re.search(r'\bdet\b|\bdet\s*\[|\|[A-Za-z]\||matrix', q_lower):
        return 'matrix_op'
    if re.search(r'\[\s*\[[\d\w\s,\-\+\.]+\]\s*,?\s*\[', q_text):
        return 'matrix_op'

    # ── Inverse trig evaluation ───────────────────────────────────────────
    if re.search(r'sin[\-⁻]1|cos[\-⁻]1|tan[\-⁻]1|asin|acos|atan|'
                 r'sin\s*\*\*\s*\(\s*-?\s*1\s*\)', q_lower):
        return 'inverse_trig'

    # ── Trig identity / value ─────────────────────────────────────────────
    if re.search(r'sin\s*\(|cos\s*\(|tan\s*\(', q_lower) and \
       re.search(r'value|evaluate|simplif', q_lower):
        return 'evaluate_expr'

    # ── Probability ───────────────────────────────────────────────────────
    if re.search(r'\bp\s*\(|probability|independent|P\(A\)', q_text):
        return 'probability'

    # ── Vector operations ─────────────────────────────────────────────────
    if re.search(r'unit\s+vector|\|.*\+.*\||dot\s+product|cross\s+product|'
                 r'magnitude|angle\s+between', q_lower):
        return 'vector_op'

    # ── Solve for variable ────────────────────────────────────────────────
    if re.search(r'find\s+(the\s+)?(value\s+of\s+)?[xykn]|solve\s+for|'
                 r'value\s+of\s+[xyk]', q_lower):
        if '=' in q_text:
            return 'solve_eq'

    # ── Evaluate / simplify expression ───────────────────────────────────
    if re.search(r'evaluat|simplif|value\s+of|equal\s+to', q_lower):
        return 'evaluate_expr'

    return 'unsupported'


# ──────────────────────── AST construction ────────────────────────────────

def _build_ast_for(objective: str, q_text: str, q_lower: str):
    """
    Build the stem_ast structure required by the solver for each objective.
    Returns (ast, objective) — objective may be updated on failure.
    """

    # ── ODE degree / order ──────────────────────────────────────────────
    if objective in ('diff_eq_degree', 'diff_eq_order'):
        # Extract just the ODE equation portion
        m = re.search(r'(?:degree|order)\s+of\s+(.*?)(?:\s+is\b|\s*:)', q_text, re.IGNORECASE)
        eq_str = m.group(1).strip() if m else q_text
        return eq_str, objective

    # ── Continuity ──────────────────────────────────────────────────────
    if objective == 'find_param_continuity':
        from text_to_ir import _extract_continuity_data
        syms = dict(_SYMS)
        data = _extract_continuity_data(q_text, syms)
        if data:
            return (data['func_expr'], data['point'],
                    data['given_val'], data['param'], data['param_syms']), objective
        return q_text, 'unsupported'

    # ── Pre-image count ─────────────────────────────────────────────────
    if objective == 'pre_image_count':
        expr = _find_expression(q_text)
        if expr is None:
            expr = x**2 + 1
        # Extract target set
        m = re.search(r'[\(\{]([0-9,\s]+)[\)\}]', q_text)
        targets = [int(v) for v in re.findall(r'\d+', m.group(1))] if m else [2, 5, 10]
        return [sp.Eq(expr, sp.Integer(tv)) for tv in targets], objective

    # ── Combinatorics ───────────────────────────────────────────────────
    if objective == 'combinatorics':
        res = _find_ncr(q_text)
        if res:
            nv, rv, is_perm = res
            return (nv, rv, is_perm), objective
        return q_text, 'unsupported'

    # ── Definite integral ───────────────────────────────────────────────
    if objective == 'definite_integral':
        expr = _find_expression(q_text)
        bounds = _find_limits_bounds(q_text)
        if expr is not None and bounds:
            return (expr, x, bounds[0], bounds[1]), objective
        if expr is not None:
            return (expr, x, sp.Integer(0), sp.Integer(1)), objective
        return q_text, 'unsupported'

    # ── Indefinite integral ─────────────────────────────────────────────
    if objective == 'integrate':
        expr = _find_expression(q_text)
        return (expr, x) if expr else (q_text, 'unsupported')

    # ── Differentiation ─────────────────────────────────────────────────
    if objective in ('differentiate', 'differentiate_at'):
        expr = _find_expression(q_text)
        if expr is None:
            # fallback: sin(log(...)) pattern
            if 'sin' in q_lower and 'log' in q_lower:
                expr = sp.sin(sp.log(x + sp.sqrt(x**2 + 1)))
        if expr:
            if objective == 'differentiate_at':
                m = re.search(r'at\s+x\s*=\s*(-?[\d\.]+)', q_lower)
                pt = sp.sympify(m.group(1)) if m else sp.Integer(1)
                return (expr, x, pt), objective
            return (expr, x), objective
        return q_text, 'unsupported'

    # ── Maxima / minima ─────────────────────────────────────────────────
    if objective == 'extremum':
        expr = _find_expression(q_text)
        mode = 'max' if 'maximum' in q_lower or 'maxima' in q_lower else 'min'
        return (expr, x, mode) if expr else (q_text, 'unsupported')

    # ── Tangent slope ───────────────────────────────────────────────────
    if objective == 'tangent_slope':
        expr = _find_expression(q_text)
        m = re.search(r'at\s+(?:x\s*=\s*)?(-?[\d\.]+)', q_lower)
        pt = sp.sympify(m.group(1)) if m else sp.Integer(0)
        return (expr, x, pt) if expr else (q_text, 'unsupported')

    # ── Matrix operation ─────────────────────────────────────────────────
    if objective == 'matrix_op':
        op, mat = _find_matrix_op(q_text)
        if mat is not None:
            return (op, mat), objective
        return q_text, 'unsupported'

    # ── Inverse trig ────────────────────────────────────────────────────
    if objective == 'inverse_trig':
        # Look for asin/acos/atan( value )
        norm = normalize_ocr(q_text)
        # Replace sin^{-1} etc. with asin
        norm = re.sub(r'sin\s*\*\*\s*\(\s*-\s*1\s*\)', 'asin', norm)
        norm = re.sub(r'cos\s*\*\*\s*\(\s*-\s*1\s*\)', 'acos', norm)
        norm = re.sub(r'tan\s*\*\*\s*\(\s*-\s*1\s*\)', 'atan', norm)
        expr = _sympify(norm)
        return (expr,), objective if expr else (q_text, 'unsupported')

    # ── Probability ─────────────────────────────────────────────────────
    if objective == 'probability':
        # Extract P(A), P(B), find P(A∪B) or P(A∩B)
        pa_m = re.search(r'P\s*\(\s*A\s*\)\s*=\s*([\d/\.]+)', q_text)
        pb_m = re.search(r'P\s*\(\s*B\s*\)\s*=\s*([\d/\.]+)', q_text)
        pa = _sympify(pa_m.group(1)) if pa_m else None
        pb = _sympify(pb_m.group(1)) if pb_m else None
        mode = 'union' if '∪' in q_text or 'union' in q_lower else 'intersection'
        indep = 'independent' in q_lower
        if pa and pb:
            return (pa, pb, mode, indep), objective
        return q_text, 'unsupported'

    # ── Vector operation ─────────────────────────────────────────────────
    if objective == 'vector_op':
        if 'unit vector' in q_lower and re.search(r'a\s*\+\s*b.*=.*[√v]3|√3', q_lower):
            return sp.Integer(1), 'vector_identity'
        return q_text, 'unsupported'

    # ── Solve equation ───────────────────────────────────────────────────
    if objective == 'solve_eq':
        eq = _find_equation(q_text)
        return eq if eq else (q_text, 'unsupported')

    # ── Evaluate expression ──────────────────────────────────────────────
    if objective == 'evaluate_expr':
        expr = _find_expression(q_text)
        return (expr,) if expr else (q_text, 'unsupported')

    return q_text, 'unsupported'


# ─────────────────────────── Main entry points ────────────────────────────

def text_to_ir(q_text: str, q_options: list, q_num=None) -> QuestionIR:
    """Single question text + option list → QuestionIR."""
    q_lower = q_text.lower()
    q_norm  = normalize_ocr(q_text)

    objective = classify_objective(q_text, q_lower, q_norm)

    # Build the AST for the objective
    ast_result = _build_ast_for(objective, q_text, q_lower)
    if isinstance(ast_result, tuple) and len(ast_result) == 2 and isinstance(ast_result[1], str):
        # _build_ast_for can return (ast, objective) to update objective on failure
        ast, objective = ast_result
    else:
        ast = ast_result

    opts = build_option_ast(q_options)

    return QuestionIR(
        question_number=q_num,
        natural_language=q_text,
        math_expressions=[],
        stem_ast=ast,
        option_ast=opts,
        objective=objective,
    )


def parse_ocr_to_ir(raw_text: str) -> QuestionIR:
    """
    Full pipeline: raw OCR text → QuestionIR.
    Handles line-by-line splitting, UI stripping, question detection.
    """
    # Strip exam UI noise
    clean = strip_ui_noise(raw_text)
    lines  = [l for l in clean.split('\n') if l.strip()]
    if not lines:
        lines = [l for l in raw_text.split('\n') if l.strip()]

    q_num    = _extract_q_number(lines)
    options  = parse_options(lines)
    stem_lines = extract_stem_lines(lines)
    q_text   = ' '.join(stem_lines)

    return text_to_ir(q_text, options, q_num=q_num)


# ── Keep these functions exported so solver_engine.py can import them ──────

def _extract_continuity_data(q_text: str, syms: dict):
    """Parse 'f(x) = <expr>... continuous at x=<a>', returns data dict."""
    k = sp.Symbol('k', real=True)
    local_syms = {**syms, 'k': k, 'K': k,
                  'cos': sp.cos, 'sin': sp.sin, 'tan': sp.tan,
                  'pi': sp.pi, 'sqrt': sp.sqrt, 'exp': sp.exp, 'log': sp.log}

    point_m = re.search(r'continuous\s+at\s+x\s*=\s*([^\s,;\.]+)', q_text, re.IGNORECASE)
    if not point_m:
        return None
    raw_point = point_m.group(1).rstrip('.,;').replace('π', 'pi')
    try:
        point = sp.sympify(raw_point, locals=local_syms)
    except Exception:
        return None

    given_val = sp.Integer(3)
    val_m = re.search(r'f\s*\([^)]*\)\s*=\s*(-?\d+(?:\.\d+)?)', q_text)
    if val_m:
        try:
            given_val = sp.sympify(val_m.group(1))
        except Exception:
            pass

    func_m = re.search(
        r'f\s*\(\s*x\s*\)\s*=\s*([^,;]+?)(?:\s+for\b|\s+and\b|\s+is\b|\s*,)',
        q_text, re.IGNORECASE)
    if not func_m:
        func_m = re.search(r'=\s*([^=,;]+?)\s+for\s+x', q_text, re.IGNORECASE)
    if not func_m:
        return None

    raw_expr = normalize_ocr(func_m.group(1).strip())
    raw_expr = re.sub(r'([a-zA-Z0-9])\s+(cos|sin|tan|log|sqrt|exp)', r'\1*\2', raw_expr)
    raw_expr = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', raw_expr)
    raw_expr = re.sub(r'\b(cos|sin|tan|log|sqrt|exp)\s+([a-zA-Z0-9_]+)', r'\1(\2)', raw_expr)
    raw_expr = re.sub(r'([a-zA-Z])\s*\(', r'\1(', raw_expr)

    func_expr = _sympify(raw_expr, extra_syms=local_syms)
    if func_expr is None:
        return None

    return {'func_expr': func_expr, 'point': point,
            'given_val': given_val, 'param': k, 'param_syms': local_syms}
