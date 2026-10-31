"""Classify a candidate expression against the corpus.

Four classes, from strongest to weakest evidence of novelty:
    NOVEL         Not equivalent to, and not a binary composition of,
                  any corpus law. This is the class to pursue.
    COMPOSITE     Expressible as law_i OP law_j for OP in {*, +, /}.
                  Interesting only if you have reason to think the
                  composition is a physical law (rare).
    SCALED_KNOWN  Constant multiple of a corpus law. Not novel — a
                  rescaled rediscovery. Filter these out.
    KNOWN         Algebraically equivalent to a corpus law (up to
                  scalar). Pure rediscovery.

IMPORTANT LIMITATION: NOVEL is a *lower bound* on novelty. It means
"not covered by the 56-law corpus." It does NOT mean "unknown to
science." Real novelty additionally requires:
    - pass litcheck with law_name=None -> ANALOGOUS or NOT_FOUND
    - survive bootstrap resampling (same form recovered on held-out data)
    - human review before publication

Composition search is restricted to 2-way (one op, two laws). 3-way
compositions will be classified NOVEL and reviewed by hand. This is
intentional: 3-way compositions of corpus laws are almost never
physically meaningful, so flagging them as NOVEL costs little.
"""
from __future__ import annotations
import math
import random
import sympy as sp

from purediscover.resource.laws import REGISTRY


def _sample_ratio_consistent(a, b, atol: float = 0.02,
                             n_trials: int = 15, seed: int = 42) -> bool:
    """True iff a/b is constant across random samples of free symbols.

    Numeric check — much faster than sympy.simplify(a/b). Fails open
    (returns False) on any evaluator exception, which is the safe
    direction: unknown does not mean "novel."
    """
    try:
        syms = list(a.free_symbols | b.free_symbols)
        if not syms:
            return False
        rng = random.Random(seed)
        ratios = []
        for _ in range(n_trials):
            sub = {s: rng.uniform(0.6, 3.0) for s in syms}
            try:
                va = float(a.subs(sub))
                vb = float(b.subs(sub))
            except (ZeroDivisionError, TypeError, ValueError):
                continue
            if abs(vb) < 1e-9:
                continue
            ratios.append(va / vb)
        if len(ratios) < n_trials // 2:
            return False
        r0 = ratios[0]
        # Underflow/overflow guard: if r0 is degenerate (all ratios
        # collapsed to ~0 or ~inf due to floating-point limits), the
        # "constant ratio" is an artefact, not a real match.
        if not math.isfinite(r0) or abs(r0) < 1e-10 or abs(r0) > 1e10:
            return False
        return all(abs(r - r0) <= atol * max(1.0, abs(r0)) for r in ratios)
    except Exception:
        return False


def _compatible(candidate_vars: set, law) -> bool:
    """A law can participate in a composition only if all its variables
    are already present in the candidate."""
    return law.expr.free_symbols.issubset(candidate_vars)


def classify_novelty(expr, corpus=None, *, n_trials: int = 15,
                     atol: float = 0.02) -> dict:
    """Classify `expr` against `corpus`.

    Returns a dict:
        {"class": "NOVEL"|"COMPOSITE"|"SCALED_KNOWN"|"KNOWN",
         "matched": law_name            (KNOWN / SCALED_KNOWN)
         "op": str, "parts": [str,str]  (COMPOSITE)}
    """
    corpus = corpus if corpus is not None else REGISTRY
    cand_vars = expr.free_symbols

    # 1. equivalence / scaled
    for law in corpus:
        if _sample_ratio_consistent(expr, law.expr, atol=atol,
                                    n_trials=n_trials):
            try:
                exact = sp.simplify(expr - law.expr) == 0
            except Exception:
                exact = False
            return {"class": "KNOWN" if exact else "SCALED_KNOWN",
                    "matched": law.name}

    # 2. binary compositions, restricted to laws with compatible vars
    compat = [l for l in corpus if _compatible(cand_vars, l)]
    n = len(compat)

    # 2a. product
    for i in range(n):
        for j in range(i, n):
            try:
                prod = compat[i].expr * compat[j].expr
            except Exception:
                continue
            if _sample_ratio_consistent(expr, prod, atol=atol,
                                        n_trials=n_trials):
                return {"class": "COMPOSITE", "op": "*",
                        "parts": [compat[i].name, compat[j].name]}

    # 2b. sum
    for i in range(n):
        for j in range(i + 1, n):
            try:
                s = compat[i].expr + compat[j].expr
            except Exception:
                continue
            if _sample_ratio_consistent(expr, s, atol=atol,
                                        n_trials=n_trials):
                return {"class": "COMPOSITE", "op": "+",
                        "parts": [compat[i].name, compat[j].name]}

    # 2c. quotient
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            try:
                q = compat[i].expr / compat[j].expr
            except Exception:
                continue
            if _sample_ratio_consistent(expr, q, atol=atol,
                                        n_trials=n_trials):
                return {"class": "COMPOSITE", "op": "/",
                        "parts": [compat[i].name, compat[j].name]}

    return {"class": "NOVEL"}


__all__ = ["classify_novelty"]
