"""Verify the convergence checker rejects surrogates and accepts true forms."""
import sympy as sp

from purediscover.benchmark.convergence import ConvergenceChecker
from purediscover.benchmark.phase_catalog import PhaseCatalog
from purediscover.benchmark.catalog import make_data


checker = ConvergenceChecker()

# --- FALSE POSITIVE 1: cow_phase surrogate ---
law = PhaseCatalog.default().get("cow_phase")
data = make_data(law, 15, 0.02)
surrogate = sp.sympify(
    "1.0278*T**2*g*m - 0.0268*T*g**2 + 0.0918 - 0.0598/g"
)
result = checker.is_converged(surrogate, law.expr, data)
print(f"cow_phase surrogate   : {result}  (expect False)")

# --- FALSE POSITIVE 2: atom_grav_gradient surrogate ---
law2 = PhaseCatalog.default().get("atom_grav_gradient")
data2 = make_data(law2, 5, 0.02)
surrogate2 = sp.sympify(
    "9*T**3/8 + 0.929*g + 0.0127*m**4 - sp.Rational(2,3)"
)
result2 = checker.is_converged(surrogate2, law2.expr, data2)
print(f"atom_grav surrogate   : {result2}  (expect False)")

# --- TRUE POSITIVE: real cow_phase ---
real = sp.sympify("T**2*g*m")
result3 = checker.is_converged(real, law.expr, data)
print(f"cow_phase REAL        : {result3}  (expect True)")

# --- TRUE POSITIVE with scalar multiple ---
real2 = sp.sympify("3.7 * T**2*g*m")
result4 = checker.is_converged(real2, law.expr, data)
print(f"cow_phase scalar mult : {result4}  (expect True)")

# --- TRUE POSITIVE for atom_grav_gradient ---
real3 = law2.expr
result5 = checker.is_converged(real3, law2.expr, data2)
print(f"atom_grav REAL        : {result5}  (expect True)")

print()
if result is False and result2 is False and result3 and result4 and result5:
    print("✓ Convergence fix working correctly")
    print("  - Rejects surrogates that fit on data but differ structurally")
    print("  - Accepts true forms (including scalar multiples)")
else:
    print("✗ Something still wrong — check individual results above")
