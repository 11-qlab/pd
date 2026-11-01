"""Full discovery pipeline: search -> classify -> rank.

v3 changelog (fixes after second real-data run):

1. SCALE CONSTANTS IN GRAMMAR
   Normalizing y -> y/mean(y) creates a target whose coefficient is
   1/mean(y). If mean(y) = 127, brute_search must produce
   I*R*(1/127) — but 1/127 isn't a leaf constant. brute_search
   reaches L7 with 200k candidates, finds nothing, returns empty.
   compose then does all the work (73s instead of 0.3s).

   Fix: pass [1.0, y_scale, 1/y_scale] as grammar constants. Target
   I*R*(1/y_scale) becomes reachable at L3 (3-4 nodes). Direct
   search finds it, compose doesn't run.

2. CANONICALIZE log(exp(x)) -> x
   sp.simplify does not reduce log(exp(R)) for symbolic R (would
   need R>0 assumption). Result: I*log(exp(R)) is functionally I*R
   but classifies as a different expression, polluting top-5 with
   duplicates and downgrading KNOWN to SCALED_KNOWN.

   Fix: explicit replacement log(exp(x)) -> x applied after
   denormalization. Safe (never fires unless the pattern is present).

3. PURE RELATIVE ERROR IN _fit_error
   v2 used abs(y - yp) / max(1.0, abs(y)). This under-reports error
   at small y (denominator floored at 1.0 = absolute error for y<1).
   v3 uses abs(y - yp) / (abs(y) + 1e-9) — consistent relative error
   across the dynamic range.

3 changes, one file. discover_and_classify signature unchanged.
"""
from __future__ import annotations
import csv
from fractions import Fraction

import sympy as sp

from purediscover.symbolic import Grammar
from purediscover.rich_search import brute_search
from purediscover.compose import search_composition, search_offset
from purediscover.novelty import classify_novelty
from purediscover.resource.laws import REGISTRY
from purediscover.exponential import search_exponential
from purediscover.sum_features import search_sum_of_features


__version__ = "4.0-power10-scale"

DEFAULT_OPS = ("add", "sub", "mul", "div", "pow", "sqrt", "log", "exp")

_NICE_DENOMS = {1, 2, 3, 4, 5, 6, 8, 10, 20, 25, 50, 100}


def _count_nodes(expr) -> int:
    return sum(1 for _ in sp.preorder_traversal(expr))


def _fit_error(expr, data, y_scale=None) -> float:
    """Scale-aware relative error. Denominator floored at data median/20."""
    if y_scale is None:
        mags = sorted(abs(y) for _, y in data if abs(y) > 0)
        y_scale = mags[len(mags) // 2] if mags else 1.0
    floor = max(y_scale / 20.0, 1e-300)
    errs = []
    for env, y in data:
        try:
            yp = float(expr.subs({sp.Symbol(k): v for k, v in env.items()}))
        except (ZeroDivisionError, ValueError, TypeError):
            return 1.0
        if not (yp == yp and abs(yp) < 1e300):
            return 1.0
        errs.append(abs(y - yp) / max(abs(y), floor))
    return sum(errs) / max(1, len(errs))


def _canonicalize(expr):
    """Reduce log(exp(x)) -> x. Symbolic-safe rewrite."""
    try:
        expr = expr.replace(
            lambda e: isinstance(e, sp.log) and isinstance(e.args[0], sp.exp),
            lambda e: e.args[0].args[0],
        )
    except Exception:
        pass
    return expr


def _simplify(expr):
    try:
        s = sp.simplify(expr)
        return _canonicalize(s if s.free_symbols else expr)
    except Exception:
        return _canonicalize(expr)


def _snap_float(fv, eps: float = 0.02):
    """Snap a float to a small rational, or return None to keep as float.

    BUG FIXED: v5's version returned Integer(0) for any |fv| < 1e-9,
    which destroyed tiny physical coefficients like hbar/2 ≈ 5.27e-35
    (Heisenberg) or the Planck prefactor 2h/c^2 ≈ 5.6e-24. Any fit
    with a legitimate tiny coefficient was being killed by snapping
    the coefficient to exactly zero.

    v6 only snaps to zero when fv IS zero. Nonzero floats either snap
    to a small rational (relative tolerance eps) or stay as floats.
    """
    if fv == 0.0:
        return sp.Integer(0)
    # Try small denominators first (1/1, 1/2, 3/4, ...). This is what
    # lets 0.99925 snap to 1. limit_denominator(1000) gives 999/1000,
    # which has a non-nice denominator. The multi-scale loop tries the
    # simplest rational first — the correct order for tolerance-based
    # snapping.
    for max_den in (10, 100, 1000):
        frac = Fraction(fv).limit_denominator(max_den)
        if frac.denominator not in _NICE_DENOMS:
            continue
        if abs(frac.numerator) > 100:
            continue
        denom = max(abs(fv), abs(float(frac)))
        if denom == 0:
            continue
        if abs(fv - float(frac)) / denom < eps:
            return sp.Rational(frac.numerator, frac.denominator)
    return None


def _snap_coefficients(expr, eps: float = 0.02):
    """Snap Floats AND ugly Rationals to small clean values.

    Single predicate used for both gate and transform. Prior versions
    had the outer gate use `q > 20` while the inner check used
    `q > 10` — a classic partial-patch bug that let a**(29/19) through
    because the outer gate never forwarded it to the inner snapper.

    Rules:
        Float    -> _snap_float (multi-scale limit_denominator)
        Rational -> if q > 10, try limit_denominator(10); snap if
                    relative error < eps
    """
    from fractions import Fraction

    def _should_visit(e):
        return e.is_Float or (e.is_Rational and e.q > 10)

    def _snap(e):
        if e.is_Float:
            r = _snap_float(float(e), eps)
            return r if r is not None else e
        if e.is_Rational and e.q > 10:
            f = float(e)
            if abs(f) < 1e-12:
                return e
            small = Fraction(f).limit_denominator(10)
            if small.denominator > 10:
                return e
            if abs(f - float(small)) / max(abs(f), 1e-9) < eps:
                return sp.Rational(small.numerator, small.denominator)
        return e

    try:
        return expr.replace(_should_visit, _snap)
    except Exception:
        return expr


def _normalize(data):
    """Scale y to O(1) using a power of 10 from the median.

    Why power-of-10 and not mean:
        Scaling by mean(y) produces 1/mean(y) as the effective target
        coefficient. For mean = 127, that is 1/127 — a fraction whose
        denominator is not in the "nice rationals" set, so it is hard
        for brute_search to reach and hard for _snap_coefficients to
        snap. Direct search then misses the target, forcing the slow
        compose fallback.

        Scaling by 10^k produces 10^-k as the target coefficient,
        which IS a nice rational (1/100, 1/10, etc.). Direct search
        finds the target at L3-L4, no compose needed.

    Case A: mean V ≈ 127, median V ≈ 50, log10(50) ≈ 1.7, round = 2,
            scale = 100. Target becomes V/100 = I*R*0.01. Findable.

    Case B: median N ≈ 0.6, log10(0.6) ≈ -0.22, round = 0, scale = 1.
            No normalization — same as before.
    """
    if not data:
        return data, 1.0
    ys = sorted(y for _, y in data)
    median = ys[len(ys) // 2]
    if abs(median) > 1e-9:
        import math
        k = round(math.log10(abs(median)))
        scale = 10.0 ** k
    else:
        scale = max(abs(y) for y in ys) or 1.0
    norm = [(env, y / scale) for env, y in data]
    return norm, scale


def _denormalize(expr_norm, y_scale):
    return _snap_coefficients(_simplify(expr_norm * y_scale))


def load_csv(path, target_col, var_cols=None):
    data = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if var_cols is None:
            var_cols = [c for c in reader.fieldnames if c != target_col]
        for row in reader:
            try:
                env = {c: float(row[c]) for c in var_cols}
                y = float(row[target_col])
            except (KeyError, ValueError):
                continue
            data.append((env, y))
    return data, list(var_cols), target_col


def _physical_score(expr) -> int:
    """Higher = more physically plausible coefficients.

    Physics laws tend to have integer exponents and small rational
    coefficients. Curve-fit surrogates (2.388/x**0.313) have Float
    atoms and ugly fractions. This is a soft tiebreak used after
    (class, n_nodes) and before (fit_error), so clean integer-coefficient
    forms win ties against slightly-better-fitting surrogates.
    """
    score = 0
    for atom in expr.atoms(sp.Float):
        score -= 2
    for atom in expr.atoms(sp.Rational):
        # Integer denominators (=1) are free; ugly fractions cost
        if atom.q > 10:
            score -= 1
    return score


def _try_extend_polynomial(expr, data, var_names, tol: float = 0.02):
    """If expr is a single-term power (c·x^n), try adding c·x^m for m > n.

    Handles partial fits where brute_search found the leading term of
    a multi-term polynomial. Example: on data from 4.2·T² + 0.08·T³,
    brute_search returns 4.28·T². This function adds the missing
    0.08·T³ term by fitting the residual.
    """
    if len(var_names) != 1:
        return None
    try:
        import numpy as np
    except ImportError:
        return None

    v = var_names[0]
    x = sp.Symbol(v)

    residuals = []
    for env, y in data:
        try:
            y_pred = float(expr.subs({x: env[v]}))
        except Exception:
            return None
        if not np.isfinite(y_pred):
            return None
        residuals.append((env, y - y_pred))

    # Detect original power so we try only HIGHER powers (T³, T⁴, T⁵)
    # Trying lower/same powers re-fits the leading term and gives a
    # bad candidate like 4.28·T² + 0.1·T² = 4.38·T² (wrong structure).
    orig_power = 1
    for atom in expr.atoms(sp.Pow):
        if len(atom.args) == 2 and atom.args[0] == x:
            try:
                p_val = float(atom.args[1])
                if p_val == int(p_val) and p_val > 0:
                    orig_power = max(orig_power, int(p_val))
            except Exception:
                pass
    if expr == x:
        orig_power = 1

    for n in (2, 3, 4, 5):
        if n <= orig_power:
            continue
        num, den = 0.0, 0.0
        for env, r in residuals:
            xn = env[v] ** n
            if not np.isfinite(xn):
                continue
            num += r * xn
            den += xn * xn
        if den < 1e-12:
            continue
        c = num / den
        try:
            candidate = expr + sp.Float(c) * x**n
            fe = _fit_error(candidate, data)
        except Exception:
            continue
        if fe <= tol:
            return candidate
    return None


def discover_and_classify(
    data, var_names, *,
    grammar_ops=DEFAULT_OPS,
    max_nodes_direct: int = 10,
    time_budget_direct: float = 30.0,
    time_budget_compose: float = 120.0,
    tol: float = 0.02,
    search_tol: float | None = None,
    corpus=None,
):
    corpus = corpus if corpus is not None else REGISTRY
    if search_tol is None:
        search_tol = max(tol * 3.0, 0.05)

    norm_data, y_scale = _normalize(data)

    # 1. direct brute on normalized data; grammar has scale constants
    g_direct = Grammar(
        variables=list(var_names),
        constants=[1.0, y_scale, 1.0 / y_scale],
        max_depth=3,
        ops=list(grammar_ops),
    )
    direct = brute_search(
        g_direct, norm_data,
        max_nodes=max_nodes_direct,
        max_per_level=50000,
        tol=search_tol,
        time_budget=time_budget_direct,
    )
    candidates = []
    for c in direct:
        e = _denormalize(c["expr"], y_scale)
        if _fit_error(e, data) <= tol:
            candidates.append(e)

    # 2. exponential — fast (~100ms), runs unconditionally
    try:
        exp_results = search_exponential(data, var_names, tol=tol)
    except Exception:
        exp_results = []
    for r in exp_results:
        e = _simplify(r["expr"])
        if _fit_error(e, data) <= tol:
            candidates.append(e)

    # 3. offset — fast (~500ms), runs unconditionally
    try:
        off_results = search_offset(data, var_names,
                                    ops=list(grammar_ops),
                                    tol=tol,
                                    time_budget=time_budget_compose / 2)
    except Exception:
        off_results = []
    for r in off_results:
        e = _simplify(r["expr"])
        if _fit_error(e, data) <= tol:
            candidates.append(e)

    # 4. sum-of-features — greedy forward selection (FAST: ~30 fits)
    try:
        sum_results = search_sum_of_features(data, var_names, tol=tol)
    except Exception:
        sum_results = []
    for r in sum_results:
        e = _simplify(r["expr"])
        if _fit_error(e, data) <= tol:
            candidates.append(e)

    # 5. compose — slow (~30-60s), only if all fast paths failed
    if not candidates:
        composed = search_composition(
            norm_data, var_names,
            ops=list(grammar_ops),
            max_nodes_b=3, max_nodes_a=8,
            tol=search_tol,
            time_budget=time_budget_compose,
            per_search_budget=0.5,
        )
        for r in composed:
            e = _denormalize(r["expr"], y_scale)
            if _fit_error(e, data) <= tol:
                candidates.append(e)

    # 2b. extend polynomial candidates with one extra term.
    # REPLACE original with extension when extension improves fit —
    # otherwise the original (fewer nodes) wins in ranking and the
    # extended form is never seen.
    extended = []
    keep = []
    for expr in candidates:
        ext = _try_extend_polynomial(expr, data, var_names, tol)
        if ext is not None:
            extended.append(ext)
            # Drop original from candidates; the extended version
            # supersedes it (better fit, complete form)
        else:
            keep.append(expr)
    candidates = keep + extended

    # 3. classify + rank
    seen: set[str] = set()
    ranked: list[dict] = []
    _rank = {"NOVEL": 3, "COMPOSITE": 2, "SCALED_KNOWN": 1, "KNOWN": 0}

    for expr in candidates:
        # Belt-and-suspenders: force snap at final output stage.
        # Guarantees display and fingerprint see clean forms
        # (a**(3/2) not a**(1309/873)).
        expr = _snap_coefficients(expr)
        try:
            key = str(sp.simplify(expr))
        except Exception:
            key = str(expr)
        if key in seen:
            continue
        seen.add(key)

        cls = classify_novelty(expr, corpus)
        n = _count_nodes(expr)
        fe = _fit_error(expr, data)
        phys = _physical_score(expr)
        ranked.append({
            "expr": expr,
            "class": cls["class"],
            "detail": cls,
            "n_nodes": n,
            "fit_error": fe,
            "_sortkey": (-_rank[cls["class"]], n, -phys,
                         round(fe, 3)),
        })

    ranked.sort(key=lambda r: r["_sortkey"])
    for r in ranked:
        del r["_sortkey"]
    return ranked


__all__ = ["discover_and_classify", "load_csv", "DEFAULT_OPS", "__version__"]
