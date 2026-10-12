import sympy as sp

x = sp.Symbol('x')
f = sp.Function('y')(x)
d1 = f.diff(x)
d2 = f.diff(x, 2)

# eq: (1 + d1**2)**(3/2) - d2 = 0
eq = (1 + d1**2)**(sp.Rational(3,2)) - d2

# Can sympy find the degree?
try:
    from sympy.solvers.ode.ode import ode_order
    print("Order:", ode_order(eq, f))
except ImportError:
    pass
    
# To find degree, we need to rationalize it: 
# (1 + d1**2)**3 = d2**2
# So degree is 2.
# SymPy's pdes/odes might have a degree function, or we can just use `sp.Poly` or `degree`.

from sympy import degree
# Let's see if degree works on the derivative
try:
    # First, make it a polynomial in highest order derivative
    order = ode_order(eq, f)
    if order:
        highest_deriv = f.diff(x, order)
        # We need to remove radicals.
        # But sympy might not do that automatically for 'degree'
        print("Testing manual rationalization...")
except Exception as e:
    print(e)
