import sympy as sp
import re

def compute_degree(eq_str):
    # Replace derivatives
    eq_str = eq_str.replace("d2y/dx2", "y2").replace("d²y/dx²", "y2")
    eq_str = eq_str.replace("dy/dx", "y1")
    eq_str = eq_str.replace("^", "**").replace("²", "**2").replace("³", "**3")
    eq_str = eq_str.replace("[", "(").replace("]", ")")
    
    # Split into left and right
    if "=" in eq_str:
        left, right = eq_str.split("=")
    else:
        left, right = eq_str, "0"
        
    y1, y2 = sp.symbols('y1 y2')
    L = sp.sympify(left, locals={'y1': y1, 'y2': y2})
    R = sp.sympify(right, locals={'y1': y1, 'y2': y2})
    
    # The highest derivative here is y2 (or y1 if y2 not present)
    if L.has(y2) or R.has(y2):
        highest = y2
    else:
        highest = y1
        
    eq = L - R
    
    # If the equation has fractional powers, we must rationalize it
    # We can use unrad() from sympy.solvers.solvers
    from sympy.solvers.solvers import unrad
    try:
        res = unrad(eq)
        if res:
            eq, _ = res
    except:
        pass
        
    # Now expand and find degree of highest
    eq = sp.expand(eq)
    degree = sp.degree(eq, gen=highest)
    return degree

s = "[1+(dy/dx)²]^(3/2) = d²y/dx²"
print(compute_degree(s))
