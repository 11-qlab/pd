"""Confirm the two bugs: _snap_float nuking tiny coefficients, and
literature corpus missing the A* prefix."""
import sympy as sp

from purediscover.novelty_gate.literature import (
    _extended_laws, algebraic_fingerprint,
)


print("BUG 1: _snap_float on a tiny coefficient")
print("-" * 55)
from purediscover.discover import _snap_float, _snap_coefficients
for v in (5.27e-35, 5.6e-24, 0.998, 8006.73, 1.0, 0.0):
    r = _snap_float(v)
    print(f"  _snap_float({v:>10.4e}) -> {r}")


print()
print("BUG 2: does arrhenius corpus entry have the A* prefix?")
print("-" * 55)
for name, expr, vars_, _desc in _extended_laws():
    if name in ("arrhenius", "inverse_proportional", "kepler_third"):
        fp = algebraic_fingerprint(expr)
        n_syms = len(expr.free_symbols)
        print(f"  {name:22s}  expr={expr}")
        print(f"      fingerprint = {fp}")
        print(f"      symbols     = {n_syms}")


print()
print("CORPUS entry expected: A*exp(-Ea/T) -> x0*exp(-x1/x2)")
print("Found expr expected:  10170056147*exp(-8006/T) -> x0*exp(-x1/x2)")
print()
found = sp.sympify("10170056147.0196*exp(-8006.72975736565/T)")
print(f"  fingerprint of found Arrhenius: {algebraic_fingerprint(found)}")
