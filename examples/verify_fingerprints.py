"""Verify fingerprints align between corpus entries and discovered forms."""
import sympy as sp
from purediscover.novelty_gate.literature import (
    algebraic_fingerprint, _extended_laws,
)

print("Fingerprint verification")
print("=" * 66)

corpus = {name: (expr, algebraic_fingerprint(expr))
          for name, expr, _, _ in _extended_laws()}

tests = [
    ("arrhenius",              "10170056147.0*exp(-8006.7/T)"),
    ("inverse_proportional",   "5.28e-35/dx"),
    ("kepler_third",           "0.99925*a**(3/2)"),
    ("planck",                 "nu**3/(exp(0.04799*nu)-1)"),
    ("planck_aggregated",      "5.6e-24*nu**3/(exp(0.04799*nu)-1)"),
    ("boltzmann_factor",       "exp(-Ea/T)"),
    ("gaussian",               "0.5*exp(-x**2)"),
]

for corpus_name, expr_str in tests:
    corpus_fp = corpus.get(corpus_name, (None, "(corpus missing)"))[1]
    found_fp  = algebraic_fingerprint(sp.sympify(expr_str))
    match = "✓" if corpus_fp == found_fp else "✗"
    print(f"\n{match}  corpus[{corpus_name}] vs found")
    print(f"     found expr    : {expr_str}")
    print(f"     found fp      : {found_fp}")
    print(f"     corpus fp     : {corpus_fp}")
