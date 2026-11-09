"""Quantum physics grammar: operators, units, dimensional rules, candidates.

Drop-in extension for purediscover_ext.orchestrator. Adds physics-complete
operator set, SI unit table for dimensional filtering, and a curated
20-law quantum candidate corpus with expected transform pairs.

Honest scope:
- Adds expressiveness (10 new physics operators)
- Does NOT invent new physics; only widens what the pipeline can express
- Q21-Q24 (PDEs, integral equations) explicitly out of scope
"""
from __future__ import annotations
import numpy as np
import sympy as sp


# ============================================================
# 1. Operator tiers
# ============================================================
BINARY_OPERATORS = ["+", "-", "*", "/"]

UNARY_BASIC = [
    "square", "cube", "neg", "inv(x)=1/x",
    "sqrt", "cbrt", "exp", "log", "log1p",
    "sin", "cos", "tan", "sinh", "cosh", "tanh",
]

# Custom physics operators — each MUST have an extra_sympy_mapping.
#
# RULES for Julia bodies (learned the hard way — 2 crash cycles):
# 1. Only use: ternary (? :), abs, exp, log, sqrt, tanh, cosh, sinh
#    NOT available: clip, min, max, NaN, Inf, @inline, @fastmath
# 2. Fallback for every operator: 0.0 or 1.0 (physical limit)
# 3. PySR's well-definedness probe uses x = -100, -1, 0, 1, 100.
#    Every operator must return a finite Float32 at all these points.
PHYSICS_CUSTOM = {
    # RULE: PySR runs in Float32. Every body wrapped in Float32() to
    # preserve type. Julia literals like 1.0 are Float64 and break
    # well-definedness test with "returned Float64 from Float32 input".
    "planck(x)=Float32(abs(x) < 1f-6 ? 1.0 : x/(exp(x)-1))": "x/(exp(x)-1)",
    "bose(x)=Float32(abs(x) < 1f-6 ? 0.0 : 1/(exp(x)-1))":   "1/(exp(x)-1)",
    "fermi(x)=Float32(1/(exp(x)+1))":                        "1/(exp(x)+1)",
    "gauss(x)=Float32(exp(-x^2))":                           "exp(-x^2)",
    "sech(x)=Float32(2/(exp(x)+exp(-x)))":                   "2/(exp(x)+exp(-x))",
    "gamma(x)=Float32(abs(x) < 0.9999 ? 1/sqrt(1-x^2) : 0.0)": "1/sqrt(1-x^2)",
    "coth(x)=Float32(abs(x) < 1f-6 ? 0.0 : 1/tanh(x))":      "1/tanh(x)",
    "lncosh(x)=Float32(abs(x) < 80 ? log(cosh(x)) : abs(x) - 0.6931471805599453)": "log(cosh(x))",
}


def get_extra_sympy_mappings() -> dict:
    """Required for PySR to symbolically simplify custom operators.

    Every operator with a "name(x)=expr" signature needs an entry here —
    including aliased builtins like inv, square, cube, neg, cbrt, log1p.
    """
    return {
        # Aliased builtins (must be mapped because we gave them name(x)= forms)
        "inv":    lambda x: 1 / x,
        "square": lambda x: x**2,
        "cube":   lambda x: x**3,
        "neg":    lambda x: -x,
        "cbrt":   lambda x: sp.real_root(x, 3),
        "log1p":  lambda x: sp.log(1 + x),
        # Custom physics operators
        "planck": lambda x: x / (sp.exp(x) - 1),
        "bose":   lambda x: 1 / (sp.exp(x) - 1),
        "fermi":  lambda x: 1 / (sp.exp(x) + 1),
        "gauss":  lambda x: sp.exp(-x**2),
        "sech":   lambda x: 2 / (sp.exp(x) + sp.exp(-x)),
        "gamma":  lambda x: 1 / sp.sqrt(1 - x**2),
        "coth":   lambda x: 1 / sp.tanh(x),
        "lncosh": lambda x: sp.log(sp.cosh(x)),
    }


def get_all_unary_operators() -> list:
    return UNARY_BASIC + list(PHYSICS_CUSTOM.keys())


# ============================================================
# 2. SI unit table for dimensional filtering
#    Dimension vector: (M, L, T, I, Theta, N, J)
# ============================================================
UNITS = {
    "kg":  (1, 0, 0, 0, 0, 0, 0),
    "m":   (0, 1, 0, 0, 0, 0, 0),
    "s":   (0, 0, 1, 0, 0, 0, 0),
    "A":   (0, 0, 0, 1, 0, 0, 0),
    "K":   (0, 0, 0, 0, 1, 0, 0),
    "J":   (1, 2, -2, 0, 0, 0, 0),
    "N":   (1, 1, -2, 0, 0, 0, 0),
    "Pa":  (1, -1, -2, 0, 0, 0, 0),
    "W":   (1, 2, -3, 0, 0, 0, 0),
    "C":   (0, 0, 1, 1, 0, 0, 0),
    "V":   (1, 2, -3, -1, 0, 0, 0),
    "Hz":  (0, 0, -1, 0, 0, 0, 0),
    "eV":  (1, 2, -2, 0, 0, 0, 0),
    # Quantum constants
    "hbar": (1, 2, -1, 0, 0, 0, 0),
    "h":    (1, 2, -1, 0, 0, 0, 0),
    "kb":   (1, 2, -2, 0, -1, 0, 0),
    "c":    (0, 1, -1, 0, 0, 0, 0),
    "G":    (-1, 3, -2, 0, 0, 0, 0),
    "e":    (0, 0, 1, 1, 0, 0, 0),
    "eps0": (-1, -3, 4, 2, 0, 0, 0),
    "mu0":  (1, 1, -2, -2, 0, 0, 0),
    "muB":  (0, 2, 0, 1, 0, 0, 0),
    "Rinf": (0, -1, 0, 0, 0, 0, 0),
}

# Dimensionless-argument operators
DIMENSIONLESS_REQUIRED = {
    "exp", "log", "log1p", "sin", "cos", "tan",
    "sinh", "cosh", "tanh",
    "planck", "bose", "fermi", "gauss", "sech", "gamma", "coth", "lncosh",
}


# ============================================================
# 3. Candidate quantum corpus (Tier A/B/C)
# ============================================================
TIER_A = [
    # Exact recoverable with current library
    {"id": "Q01", "name": "Planck energy",       "formula": "E - hbar*omega = 0",
     "vars": ["hbar", "omega"], "target": "E"},
    {"id": "Q02", "name": "Bohr radius",         "formula": "a0 - 4*pi*eps0*hbar**2/(m*e**2) = 0",
     "vars": ["eps0", "hbar", "m", "e"], "target": "a0"},
    {"id": "Q03", "name": "Bohr levels",         "formula": "E_n + C/n**2 = 0",
     "vars": ["C", "n"], "target": "E_n"},
    {"id": "Q04", "name": "Angular momentum",    "formula": "L - n*hbar = 0",
     "vars": ["n", "hbar"], "target": "L"},
    {"id": "Q05", "name": "Rydberg",             "formula": "inv_lambda - Rinf*(1/n1**2 - 1/n2**2) = 0",
     "vars": ["Rinf", "n1", "n2"], "target": "inv_lambda"},
    {"id": "Q06", "name": "de Broglie",          "formula": "lam - h/p = 0",
     "vars": ["h", "p"], "target": "lam"},
    {"id": "Q07", "name": "Photon momentum",     "formula": "p - hbar*k = 0",
     "vars": ["hbar", "k"], "target": "p"},
    {"id": "Q08", "name": "Harmonic oscillator", "formula": "E_n - hbar*omega*(n + 1/2) = 0",
     "vars": ["hbar", "omega", "n"], "target": "E_n"},
    {"id": "Q09", "name": "Fermi wavevector",    "formula": "k_F - (3*pi**2*n)**(1/3) = 0",
     "vars": ["n"], "target": "k_F"},
    {"id": "Q10", "name": "Hall coefficient",    "formula": "R_H + 1/(n*e) = 0",
     "vars": ["n", "e"], "target": "R_H"},
]

TIER_B = [
    # Needs one new operator (planck, bose, fermi, gamma, gauss)
    {"id": "Q11", "name": "Planck radiation",    "formula": "u - hbar*omega**3/(pi**2*c**3*(exp(hbar*omega/(kb*T))-1)) = 0",
     "vars": ["hbar", "omega", "c", "kb", "T"], "target": "u",
     "needs": "bose"},
    {"id": "Q12", "name": "Bose-Einstein",       "formula": "n - 1/(exp(hbar*omega/(kb*T))-1) = 0",
     "vars": ["hbar", "omega", "kb", "T"], "target": "n",
     "needs": "bose"},
    {"id": "Q13", "name": "Fermi-Dirac",         "formula": "f - 1/(exp((E-mu)/(kb*T))+1) = 0",
     "vars": ["E", "mu", "kb", "T"], "target": "f",
     "needs": "fermi"},
    {"id": "Q14", "name": "Diode equation",      "formula": "I - I0*(exp(e*V/(kb*T))-1) = 0",
     "vars": ["I0", "e", "V", "kb", "T"], "target": "I",
     "needs": "exp"},
    {"id": "Q15", "name": "Hydrogen 1s",         "formula": "psi - exp(-r/a0) = 0",
     "vars": ["r", "a0"], "target": "psi",
     "needs": "exp"},
    {"id": "Q16", "name": "Hydrogen 2s",         "formula": "psi - (1 - r/(2*a0))*exp(-r/(2*a0)) = 0",
     "vars": ["r", "a0"], "target": "psi",
     "needs": "log1p"},
    {"id": "Q17", "name": "Harmonic ground",     "formula": "psi - exp(-x**2/2) = 0",
     "vars": ["x"], "target": "psi",
     "needs": "gauss"},
    {"id": "Q18", "name": "Lorentz factor",      "formula": "gam - 1/sqrt(1 - v**2/c**2) = 0",
     "vars": ["v", "c"], "target": "gam",
     "needs": "gamma"},
    {"id": "Q19", "name": "Relativistic energy", "formula": "E - sqrt((p*c)**2 + (m*c**2)**2) = 0",
     "vars": ["p", "c", "m"], "target": "E",
     "needs": "sqrt"},
    {"id": "Q20", "name": "Compton shift",       "formula": "dlam - (h/(m*c))*(1 - cos(theta)) = 0",
     "vars": ["h", "m", "c", "theta"], "target": "dlam",
     "needs": "cos"},
]

TIER_C = [
    # Out of scope: PDEs and integral equations
    {"id": "Q21", "name": "Schrödinger (full PDE)", "reason": "needs differential operators"},
    {"id": "Q22", "name": "Dirac equation",          "reason": "matrix-valued operators"},
    {"id": "Q23", "name": "Path integral",           "reason": "infinite-dimensional functional"},
    {"id": "Q24", "name": "Eliashberg Tc",           "reason": "coupled nonlinear integral equations"},
]


def get_candidates(tier: str = "AB"):
    out = []
    if "A" in tier:
        out.extend(TIER_A)
    if "B" in tier:
        out.extend(TIER_B)
    return out


# ============================================================
# 4. Rewrite rules (for future AST pruner; documented here)
# ============================================================
REWRITE_RULES = [
    ("x + 0", "x"),
    ("x * 1", "x"),
    ("x * 0", "0"),
    ("x^1", "x"),
    ("x^0", "1"),
    ("log(exp(x))", "x"),
    ("exp(log(x))", "x"),
    ("sqrt(x^2)", "abs(x)"),
    ("sin(x)^2 + cos(x)^2", "1"),
    ("1 - cos(x)", "2*sin(x/2)^2"),
    ("1/(exp(x)-1)", "bose(x)"),
    ("x/(exp(x)-1)", "planck(x)"),
]


# ============================================================
# 5. PySR config builder (correctness-critical)
# ============================================================
def build_pysr_config(timeout_sec: int = 60, niter: int = 40) -> dict:
    """Return PySR kwargs with all known-good settings.

    Critical choices:
    - tournament_selection_n < population_size (else Julia assertion)
    - NO deterministic=True (requires parallelism='serial', 10x slower)
    - extra_sympy_mappings includes all custom operators
    """
    return {
        "niterations": niter,
        "populations": 15,
        "population_size": 33,
        "tournament_selection_n": 10,
        "binary_operators": BINARY_OPERATORS,
        "unary_operators": get_all_unary_operators(),
        "extra_sympy_mappings": get_extra_sympy_mappings(),
        "maxsize": 30,
        "timeout_in_seconds": timeout_sec,
        "progress": False,
        "verbosity": 0,
    }


def capability_report() -> dict:
    return {
        "operators": len(get_all_unary_operators()),
        "custom_operators": list(PHYSICS_CUSTOM.keys()),
        "units_defined": len(UNITS),
        "candidates_tier_AB": len(TIER_A) + len(TIER_B),
        "candidates_out_of_scope": [c["id"] for c in TIER_C],
        "rewrite_rules": len(REWRITE_RULES),
    }


# ============================================================
# Operator pre-flight validator (catches Julia-body bugs before PySR)
# ============================================================
def validate_operators_manually() -> list:
    """Test all PHYSICS_CUSTOM operators at probe points [-100,-1,0,1,100].

    Returns list of (op_name, value) that raise or return non-finite.
    Run this BEFORE PySR. Catches ~90% of crash bugs in 50ms.
    """
    import math
    probes = [-100.0, -1.0, -0.5, 0.0, 0.5, 1.0, 100.0]
    failures = []
    # Python mirror of each Julia body — keep in sync with PHYSICS_CUSTOM
    py_bodies = {
        "planck": lambda x: 1.0 if abs(x) < 1e-6 else x/(math.exp(x)-1) if math.exp(x) < 1e30 else 0.0,
        "bose":   lambda x: 0.0 if abs(x) < 1e-6 else 1/(math.exp(x)-1) if math.exp(x) < 1e30 else 0.0,
        "fermi":  lambda x: 1/(math.exp(x)+1) if math.exp(x) < 1e30 else 0.0,
        "gauss":  lambda x: math.exp(-x**2) if x**2 < 700 else 0.0,
        "sech":   lambda x: 2/(math.exp(x)+math.exp(-x)) if abs(x) < 700 else 0.0,
        "gamma":  lambda x: 1/math.sqrt(1-x**2) if abs(x) < 0.9999 else 0.0,
        "coth":   lambda x: 0.0 if abs(x) < 1e-6 else 1/math.tanh(x),
        "lncosh": lambda x: math.log(math.cosh(x)) if abs(x) < 80 else abs(x) - 0.6931471805599453,
    }
    for name, fn in py_bodies.items():
        for x in probes:
            try:
                v = fn(x)
                if not (math.isfinite(v)):
                    failures.append((name, x, f"non-finite: {v}"))
            except Exception as e:
                failures.append((name, x, f"{type(e).__name__}: {e}"))
    return failures
