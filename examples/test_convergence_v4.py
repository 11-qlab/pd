from purediscover.benchmark.convergence import ConvergenceChecker
from purediscover.benchmark.catalog import make_data
from purediscover.benchmark.phase_catalog import PhaseCatalog
import sympy as sp

c = ConvergenceChecker()
m, g, T, x = sp.symbols("m g T x")

print("=== TRUE POSITIVES (must pass) ===")
cow_law = PhaseCatalog.default().get("cow_phase")
cow_data = make_data(cow_law, 15, 0.02)
print("exact cow  :", c.is_converged(T**2 * g * m, T**2 * g * m, cow_data))
print("scaled cow :", c.is_converged(3.7 * T**2 * g * m, T**2 * g * m, cow_data))

# Noisy fit of arrhenius (previously failed with fixed corners)
arr_law = PhaseCatalog.default().get if False else None
from purediscover.benchmark.catalog import LawCatalog
arr = LawCatalog.default().get("arrhenius")
arr_data = make_data(arr, 15, 0.02)
fit_arr = sp.Float(1.01e10) * sp.exp(sp.Float(-8005) / T)
print("noisy arr  :", c.is_converged(fit_arr, arr.expr, arr_data))  # KEY

linear_data = [({"x": xv}, 2.0 * xv) for xv in [0.5, 1.0, 2.0, 5.0, 10.0]]
print("noisy 2x   :", c.is_converged(
    sp.Float(1.9998) * x + sp.Float(0.0001),
    sp.Float(2.0) * x,
    linear_data,
))

print()
print("=== FALSE POSITIVES (must fail) ===")
cow_surr = sp.sympify("1.0278*T**2*g*m - 0.0268*T*g**2 + 0.0918 - 0.0598/g")
print("cow surr   :", c.is_converged(cow_surr, T**2 * g * m, cow_data))

ag_law = PhaseCatalog.default().get("atom_grav_gradient")
ag_data = make_data(ag_law, 5, 0.02)
ag_surr = (
    sp.Rational(9, 8) * T**3
    + sp.Rational(929, 1000) * g
    + sp.Rational(127, 10000) * m**4
    - sp.Rational(2, 3)
)
print("atom surr  :", c.is_converged(ag_surr, ag_law.expr, ag_data))
