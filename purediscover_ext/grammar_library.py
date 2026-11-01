"""Real symbolic-regression library with prior weights for MDL v2.

Each primitive has:
  fn:    callable
  arity: 1, 2, or "var" (variadic)
  prior: probability mass; must sum to <= 1 across library
  group: elementary / trig / exp / special / rational

MDL v2 can use -log2(prior(p)) as structural cost per node, which
concentrates effective complexity on common primitives and lets rare ones
win only when fit gain justifies the extra ~13 bits.

Library is curated. Adding a primitive increases L; MDL's log2(L) cost
requires proportionally more data to rank correctly (see REPORT.md).
"""
from __future__ import annotations
import numpy as np
import sympy as sp

PI = float(np.pi)


def _safe(f, fill=np.nan, clip=1e-30):
    def g(*a):
        with np.errstate(all="ignore"):
            try:
                v = f(*a)
            except Exception:
                return fill
            if isinstance(v, complex):
                return fill
            if not np.isfinite(v):
                return fill
            return v
    return g


# Each: name -> {fn, arity, prior, group, sympy_fn (for symbolic use)}
PRIMITIVES = {
    # ---- elementary ----
    "add": dict(fn=lambda a, b: a + b, arity=2, prior=0.15, group="elementary",
                sympy=lambda a, b: a + b),
    "sub": dict(fn=lambda a, b: a - b, arity=2, prior=0.08, group="elementary",
                sympy=lambda a, b: a - b),
    "mul": dict(fn=lambda a, b: a * b, arity=2, prior=0.15, group="elementary",
                sympy=lambda a, b: a * b),
    "div": dict(fn=_safe(lambda a, b: a / b if abs(b) > 1e-30 else np.nan),
                arity=2, prior=0.07, group="elementary",
                sympy=lambda a, b: a / b),
    "pow": dict(fn=_safe(lambda a, b: a ** b), arity=2, prior=0.03,
                group="elementary", sympy=lambda a, b: a ** b),
    "neg": dict(fn=lambda a: -a, arity=1, prior=0.02, group="elementary",
                sympy=lambda a: -a),

    # ---- roots / powers ----
    "sqrt": dict(fn=_safe(lambda a: np.sqrt(a) if a >= 0 else np.nan),
                 arity=1, prior=0.05, group="roots",
                 sympy=lambda a: sp.sqrt(a)),
    "inv_sqrt": dict(fn=_safe(lambda a: 1.0 / np.sqrt(a) if a > 0 else np.nan),
                     arity=1, prior=0.02, group="roots",
                     sympy=lambda a: 1 / sp.sqrt(a)),
    "sq": dict(fn=lambda a: a * a, arity=1, prior=0.02, group="roots",
               sympy=lambda a: a ** 2),

    # ---- exp / log ----
    "exp": dict(fn=_safe(np.exp), arity=1, prior=0.03, group="exp/log",
                sympy=sp.exp),
    "log": dict(fn=_safe(lambda a: np.log(a) if a > 0 else np.nan),
                arity=1, prior=0.03, group="exp/log", sympy=sp.log),
    "log10": dict(fn=_safe(lambda a: np.log10(a) if a > 0 else np.nan),
                  arity=1, prior=0.02, group="exp/log",
                  sympy=lambda a: sp.log(a, 10)),
    "log1p": dict(fn=_safe(lambda a: np.log1p(a) if a > -1 else np.nan),
                  arity=1, prior=0.01, group="exp/log", sympy=lambda a: sp.log(1 + a)),
    "expm1": dict(fn=_safe(np.expm1), arity=1, prior=0.01, group="exp/log",
                  sympy=lambda a: sp.exp(a) - 1),

    # ---- trig ----
    "sin": dict(fn=np.sin, arity=1, prior=0.03, group="trig", sympy=sp.sin),
    "cos": dict(fn=np.cos, arity=1, prior=0.03, group="trig", sympy=sp.cos),
    "tan": dict(fn=_safe(np.tan), arity=1, prior=0.005, group="trig",
                sympy=sp.tan),
    "asin": dict(fn=_safe(lambda a: np.arcsin(a) if abs(a) <= 1 else np.nan),
                 arity=1, prior=0.003, group="trig", sympy=sp.asin),
    "acos": dict(fn=_safe(lambda a: np.arccos(a) if abs(a) <= 1 else np.nan),
                 arity=1, prior=0.003, group="trig", sympy=sp.acos),
    "atan": dict(fn=np.arctan, arity=1, prior=0.005, group="trig",
                 sympy=sp.atan),

    # ---- hyperbolics / saturating ----
    "sinh": dict(fn=_safe(np.sinh), arity=1, prior=0.005, group="hyperbolic",
                 sympy=sp.sinh),
    "cosh": dict(fn=_safe(np.cosh), arity=1, prior=0.005, group="hyperbolic",
                 sympy=sp.cosh),
    "tanh": dict(fn=np.tanh, arity=1, prior=0.01, group="hyperbolic",
                 sympy=sp.tanh),
    "logistic": dict(fn=_safe(lambda a: 1.0 / (1.0 + np.exp(-a))),
                     arity=1, prior=0.01, group="hyperbolic",
                     sympy=lambda a: 1 / (1 + sp.exp(-a))),

    # ---- rational / saturating composites ----
    "recip": dict(fn=_safe(lambda a: 1.0 / a if abs(a) > 1e-30 else np.nan),
                  arity=1, prior=0.03, group="rational",
                  sympy=lambda a: 1 / a),
    "hill1": dict(fn=_safe(lambda a: a / (1.0 + a)),
                  arity=1, prior=0.01, group="rational",
                  sympy=lambda a: a / (1 + a)),

    # ---- Gaussian ----
    "gauss": dict(fn=_safe(lambda a: np.exp(-a * a)), arity=1, prior=0.01,
                  group="gauss", sympy=lambda a: sp.exp(-a ** 2)),

    # ---- special ----
    "erf": dict(fn=_safe(sp.erf, fill=0.0), arity=1, prior=0.005,
                group="special", sympy=sp.erf),
    "erfc": dict(fn=_safe(sp.erfc, fill=0.0), arity=1, prior=0.003,
                 group="special", sympy=sp.erfc),
    "gamma": dict(fn=_safe(sp.gamma), arity=1, prior=0.002,
                  group="special", sympy=sp.gamma),
    "digamma": dict(fn=_safe(sp.digamma), arity=1, prior=0.002,
                    group="special", sympy=sp.digamma),
    "zeta": dict(fn=_safe(sp.zeta), arity=1, prior=0.001,
                 group="special", sympy=sp.zeta),
    "besselj0": dict(fn=_safe(sp.besselj, fill=0.0), arity=1, prior=0.002,
                     group="special", sympy=lambda a: sp.besselj(0, a)),
}


def total_prior() -> float:
    return float(sum(p["prior"] for p in PRIMITIVES.values()))


def cost_bits(name: str) -> float:
    """Structural cost per node in bits (-log2 prior)."""
    import math
    return -math.log2(PRIMITIVES[name]["prior"])


def to_sympy(name: str, *args):
    return PRIMITIVES[name]["sympy"](*args)


def group_summary():
    from collections import defaultdict
    out = defaultdict(list)
    for name, p in PRIMITIVES.items():
        out[p["group"]].append(name)
    return dict(out)
