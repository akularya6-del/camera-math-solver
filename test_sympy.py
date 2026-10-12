import sympy as sp
x = sp.symbols('x')
print(sp.simplify(sp.sqrt(x**2) - sp.Abs(x)))
