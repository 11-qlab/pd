"""Compositional search: find y = A op B by peeling B, then searching A.

Why this exists:
    brute_search does beam-search-by-fit. It prunes candidates that
    don't fit the *full* data. For a target like m*v**2/(r+1), the
    subtree m*v**2 alone fits poorly (off by r+1), so it gets pruned
    at depth 5, before the div/add wrapper at depth 9 can form.
    Result: brute_search misses the target entirely.

    Compositional search avoids this by:
        1. Enumerate small B candidates unconditionally (not fit-pruned)
        2. For each B and each op, transform data: A = peel_op(y, B)
        3. Search for A on the transformed data
        4. Recombine: candidate = A op B

    Because A and B are searched against *different* datasets, no
    intermediate pruning happens. The composed expression can be much
    deeper than brute_search's depth limit.

Cost:
    ~20 B candidates * 4 operators = ~80 searches. Each search is
    small (A tends to be shallow). Total ~10-40s.

Example that works:
    y = m*v**2 / (r+1)
    B = "r+1" (3 nodes, in enumeration)
    op = "/" -> transformed data A = y*(r+1) = m*v**2 (5 nodes)
    brute_search finds m*v**2 in 0.1s
    composed = m*v**2 / (r+1)  ✓
"""
from __future__ import annotations
import time

import sympy as sp

from purediscover.rich_search import brute_search


def _count_nodes(expr) -> int:
    return sum(1 for _ in sp.preorder_traversal(expr))


def _enumerate_b(var_names, max_nodes: int = 3) -> list[sp.Expr]:
    """Small expressions usable as the 'B' half of a composition.

    Deliberately restricted: variables, single-op transforms, one
    two-operand sum/product. Enough to catch most denominators,
    denominators of kinetic-style ratios, and phase factors.
    """
    vars_ = [sp.Symbol(v) for v in var_names]
    small_consts = [sp.Integer(1), sp.Integer(2)]
    out: list[sp.Expr] = []
    seen: set[str] = set()

    def _add(e):
        if e is None:
            return
        try:
            if _count_nodes(e) > max_nodes:
                return
            key = str(sp.simplify(e))
        except Exception:
            return
        if key in seen:
            return
        seen.add(key)
        out.append(e)

    # leaves
    for v in vars_:
        _add(v)
    for c in small_consts:
        _add(c)

    # var * var products — REORDERED TO FRONT. In physical laws like
    # m*g*T^2, m*g is the correct peel candidate. Moving products
    # early ensures they're tried within the search budget instead of
    # being at position 40+ in the enumeration.
    for a in vars_:
        for b in vars_:
            if str(a) == str(b):
                continue
            _add(a * b)

    # unary on a variable
    for v in vars_:
        _add(v ** 2)
        _add(sp.sqrt(v))
        _add(sp.log(v))
        _add(sp.exp(v))

    # var op const
    for v in vars_:
        for c in small_consts:
            _add(v + c)
            _add(v - c)
            _add(v * c)
            _add(v / c)

    # var op var (non-product)
    for a in vars_:
        for b in vars_:
            if str(a) == str(b):
                continue
            _add(a + b)
            _add(a - b)
            _add(a / b)

    return out


def _peel(y: float, b_val: float, op: str):
    """Inverse-transform one data point: solve A from y = A op B."""
    if op == "*":
        if abs(b_val) < 1e-9:
            return None
        return y / b_val
    if op == "/":
        return y * b_val
    if op == "+":
        return y - b_val
    if op == "-":
        return y + b_val
    return None


def _recompose(A, B, op: str):
    try:
        if op == "*":
            return A * B
        if op == "/":
            return A / B
        if op == "+":
            return A + B
        if op == "-":
            return A - B
    except Exception:
        return None


def _fit_error(expr, data, y_scale=None) -> float:
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


def search_composition(
    data,
    var_names,
    *,
    ops: list[str] | None = None,
    max_nodes_b: int = 3,
    max_nodes_a: int = 8,
    operators: tuple[str, ...] = ("*", "/", "+", "-"),
    tol: float = 0.02,
    time_budget: float = 60.0,
    per_search_budget: float = 2.0,
) -> list[dict]:
    """Enumerate compositions y = A op B and return ranked candidates.

    Returns list of dicts sorted by node count (simplest first):
        {"expr": composed sympy expression,
         "decomposition": (str(A), op, str(B)),
         "fit_error": float,
         "n_nodes": int}
    """
    from purediscover.symbolic import Grammar

    ops = ops or ["add", "sub", "mul", "div", "pow", "sqrt", "log", "exp"]
    grammar_a = Grammar(
        variables=list(var_names),
        constants=[1.0],
        max_depth=3,
        ops=ops,
    )

    bs = _enumerate_b(var_names, max_nodes=max_nodes_b)
    print(f"  compose: {len(bs)} B candidates, "
          f"up to {len(bs) * len(operators)} searches")

    t0 = time.time()
    results: list[dict] = []
    seen_exprs: set[str] = set()
    n_searches = 0
    n_hit = 0

    for B in bs:
        if time.time() - t0 > time_budget:
            break

        # Precompute B values at data points
        b_vals = []
        skip_b = False
        for env, _ in data:
            try:
                bv = float(B.subs({sp.Symbol(k): v for k, v in env.items()}))
            except (ZeroDivisionError, ValueError, TypeError):
                skip_b = True
                break
            b_vals.append(bv)
        if skip_b:
            continue

        for op in operators:
            if time.time() - t0 > time_budget:
                break

            # Transform data: solve for A
            transformed = []
            for (env, y), bv in zip(data, b_vals):
                av = _peel(y, bv, op)
                if av is None:
                    transformed = []
                    break
                transformed.append((env, av))
            if not transformed:
                continue

            n_searches += 1
            cands = brute_search(
                grammar_a, transformed,
                max_nodes=max_nodes_a,
                max_per_level=5000,
                tol=tol,
                time_budget=per_search_budget,
            )

            def _try_constant(A_val):
                """Try A_val as a scalar constant for this (B, op)."""
                nonlocal n_hit
                composed = _recompose(sp.Float(A_val), B, op)
                if composed is None:
                    return
                try:
                    key = str(sp.simplify(composed))
                except Exception:
                    key = str(composed)
                if key in seen_exprs:
                    return
                fe = _fit_error(composed, data)
                if fe > tol:
                    return
                seen_exprs.add(key)
                n_hit += 1
                results.append({
                    "expr": composed,
                    "decomposition": (repr(A_val), op, str(B)),
                    "fit_error": fe,
                    "n_nodes": _count_nodes(composed),
                })

            # scalar-mean path — handles A/B where A is a difficult float
            ys_t = [y for _, y in transformed]
            if ys_t:
                mean_y = sum(ys_t) / len(ys_t)
                if abs(mean_y) > 1e-300:
                    spread = (max(ys_t) - min(ys_t)) if len(ys_t) > 1 else 0.0
                    if spread / abs(mean_y) < 0.5:
                        _try_constant(mean_y)

            for c in cands[:5]:
                A = c["expr"]
                composed = _recompose(A, B, op)
                if composed is None:
                    continue
                try:
                    key = str(sp.simplify(composed))
                except Exception:
                    key = str(composed)
                if key in seen_exprs:
                    continue
                fe = _fit_error(composed, data)
                if fe > tol:
                    continue
                seen_exprs.add(key)
                n_hit += 1
                results.append({
                    "expr": composed,
                    "decomposition": (str(A), op, str(B)),
                    "fit_error": fe,
                    "n_nodes": _count_nodes(composed),
                })

    results.sort(key=lambda r: (r["n_nodes"], r["fit_error"]))
    print(f"  compose: {n_searches} searches, {n_hit} unique fits, "
          f"{time.time()-t0:.1f}s")
    return results


__all__ = ["search_composition", "_enumerate_b", "_count_nodes",
    "search_power_law",
]

# --- Offset peel (y = A(x) + B) -----------------------------------------
#
# Detects additive offsets via Laurent regression on y = c0 + c1/x + c2/x².
# For data of shape y = A/x + B, the Laurent fit recovers c0 = B exactly.
# For y = A/x (no offset), c0 = 0.
# For y = A*exp(-kx), the Laurent fit underfits (residual > tol) → None.
#
# Why Laurent and not min(y)/median(y): data-driven statistics under
# estimate the offset because y(x) doesn't reach its asymptote within
# the sampled x range. A Laurent fit uses multiple x values to
# extrapolate the asymptote, which is what actually recovers B.



def _laurent_offset(data, var_names, *, max_resid: float = 0.025):
    """Return offset B for y = c0 + c1/x, or None.

    Uses 2-parameter Laurent basis (c0 + c1/x), which is the EXACT
    functional form of A/x + B. The 3-parameter variant (adding c2/x²)
    is rejected because it overfits non-Laurent shapes: a sqrt curve
    fits 3-term Laurent to residual 0.024 (passing a 0.025 threshold)
    while only 2-term Laurent correctly rejects it (residual ~0.5).

    For any data whose underlying law is genuinely A/x + B, the 2-param
    fit is exact (up to noise). For any other shape (sqrt, log, exp,
    linear, polynomial), it fails badly — which is exactly the behavior
    we want.

    Pipeline:
        1. Validate: single variable, positive x, >=5 points.
        2. Fit [1, 1/x] to data. If residual > max_resid, reject.
        3. Stability: refit c0 on low-x and high-x halves. If c0_lo
           and c0_hi disagree by > 50%, reject (noise-dominated fit).
        4. Endpoint check: fit must be tight at both x extremes.
    """
    if len(var_names) != 1:
        return None
    try:
        import numpy as np
    except ImportError:
        return None

    v = var_names[0]
    xs, ys = [], []
    for env, y in data:
        x = env.get(v)
        if x is None or x <= 0:
            return None
        xs.append(float(x))
        ys.append(float(y))
    if len(xs) < 5:
        return None

    x_arr = np.array(xs, dtype=float)
    y_arr = np.array(ys, dtype=float)

    # 2-parameter Laurent basis: {1, 1/x}
    X = np.column_stack([
        np.ones(len(xs)),
        1.0 / x_arr,
    ])
    try:
        coefs, *_ = np.linalg.lstsq(X, y_arr, rcond=None)
    except Exception:
        return None

    y_pred = X @ coefs
    resid = float(np.mean(np.abs(y_arr - y_pred) / (np.abs(y_arr) + 1e-9)))
    if resid > max_resid:
        return None

    B = float(coefs[0])

    # Stability: refit c0 on halves of the log-x range
    x_min, x_max = float(x_arr.min()), float(x_arr.max())
    if x_min > 0 and x_max / x_min > 5:
        x_mid = float(np.sqrt(x_min * x_max))
        mask_lo = x_arr <= x_mid
        mask_hi = x_arr > x_mid
        if mask_lo.sum() >= 3 and mask_hi.sum() >= 3:
            try:
                cm_lo, *_ = np.linalg.lstsq(X[mask_lo], y_arr[mask_lo], rcond=None)
                cm_hi, *_ = np.linalg.lstsq(X[mask_hi], y_arr[mask_hi], rcond=None)
                B_lo = float(cm_lo[0])
                B_hi = float(cm_hi[0])
                scale = max(abs(B_lo), abs(B_hi), 1e-9)
                if abs(B_lo - B_hi) > 0.5 * scale:
                    return None
            except Exception:
                pass

    # Endpoint residuals — belt and suspenders
    n_end = max(3, len(x_arr) // 10)
    order = np.argsort(x_arr)
    low_idx = order[:n_end]
    high_idx = order[-n_end:]

    resid_low = float(np.mean(
        np.abs(y_arr[low_idx] - y_pred[low_idx])
        / (np.abs(y_arr[low_idx]) + 1e-9)
    ))
    resid_high = float(np.mean(
        np.abs(y_arr[high_idx] - y_pred[high_idx])
        / (np.abs(y_arr[high_idx]) + 1e-9)
    ))
    if resid_low > 0.10 or resid_high > 0.10:
        return None

    return B


def search_offset(data, var_names, *, ops=None, tol: float = 0.02,
                  time_budget: float = 30.0):
    """Find y = A(x) + B via Laurent offset estimation + scalar refit.

    Pipeline:
        1. _laurent_offset estimates B from the Laurent model
        2. Data is normalized: y' = (y - B) / residual_scale
        3. brute_search finds the shape A'(x) on unit-scale data
        4. Scalar refit: c = argmin ||y' - c*A'(x)||
        5. Reconstruct: y = c * A'(x) * residual_scale + B

    Step 4 is essential: brute_search emits integer-coefficient forms
    like 1/x, but the true coefficient on normalized data is 0.3977/x.
    Without refit, the reconstruction is off by 2.5× and the fit fails.
    """
    from purediscover.symbolic import Grammar

    if not data or len(data) < 5:
        return []

    B = _laurent_offset(data, var_names)
    if B is None:
        return []

    residuals = [abs(y - B) for _, y in data]
    residual_scale = max(residuals) if residuals else 1.0
    if residual_scale < 1e-9:
        return []

    transformed = [(env, (y - B) / residual_scale) for env, y in data]

    g = Grammar(
        variables=list(var_names),
        constants=[1.0],
        max_depth=3,
        ops=list(ops or ["add", "sub", "mul", "div", "pow",
                          "sqrt", "log", "exp"]),
    )

    search_tol = max(tol * 10.0, 0.5)
    cands = brute_search(
        g, transformed,
        max_nodes=8,
        max_per_level=50000,
        tol=search_tol,
        time_budget=time_budget,
    )

    def _refit_scalar(form):
        """Least-squares fit of c in y_trans = c * form(x)."""
        import math
        num, den = 0.0, 0.0
        for env, y_t in transformed:
            try:
                fv = float(form.subs({sp.Symbol(k): v for k, v in env.items()}))
            except Exception:
                continue
            if not math.isfinite(fv):
                continue
            num += y_t * fv
            den += fv * fv
        if den < 1e-12:
            return None
        return num / den

    out = []
    for c in cands:
        try:
            form = c["expr"]
            scalar = _refit_scalar(form)
            if scalar is None:
                continue
            composed = sp.Float(scalar) * form * residual_scale + sp.Float(B)
            composed = sp.simplify(composed)
        except Exception:
            continue
        fe = _fit_error(composed, data)
        if fe > tol:
            continue
        out.append({
            "expr": composed,
            "offset": B,
            "fit_error": fe,
            "n_nodes": _count_nodes(composed),
        })
    return sorted(out, key=lambda r: (r["n_nodes"], r["fit_error"]))


__all__ = [name for name in globals() if not name.startswith("_")] + [
    "_laurent_offset", "search_offset",
]


# =========================================================================
# FINAL OVERRIDE — appended at file end so this wins over any earlier def.
# Prior patches left duplicate _laurent_offset definitions in the file
# (from incremental edits), and Python uses the last one. This wrapper
# applies strict checks regardless of what the wrapped version does.
# =========================================================================

# --- Power-law fit: log-linear regression for arbitrary coefficients ---
#
# Complements search_offset for the case B = 0 (pure A·x^k). Grammar-based
# search cannot represent arbitrary float coefficients like ħ/2 ≈ 5.27e-35
# because the grammar constants are integers or simple powers. Log-linear
# regression fits A and k directly as floats, catching:
#
#     Heisenberg:      dp = ħ/(2·dx)      → A ≈ 5.27e-35, k = -1
#     Stefan-Boltzmann: J = σ·T⁴          → A = σ, k = 4
#     Kepler III:      P = a^(3/2)        → A = 1, k = 1.5
#     Coulomb:         F = k·q1·q2/r²     → multi-var, coefficients per var
#
# Returns [] if fit error exceeds tol. For exponential laws (Arrhenius),
# log-fit fails because log(y) is not linear in log(x).

def search_power_law(data, var_names, *, tol: float = 0.02):
    """Log-linear fit for y = A·∏ x_i^k_i. Handles arbitrary float A, k."""
    if len(data) < 3:
        return []

    # Validate positivity
    for env, y in data:
        if y <= 0:
            return []
        for v in var_names:
            xv = env.get(v)
            if xv is None or xv <= 0:
                return []

    try:
        import numpy as np
    except ImportError:
        return []

    cols = [np.log([env[v] for env, _ in data]) for v in var_names]
    log_y = np.log([y for _, y in data])
    X = np.column_stack([np.ones(len(data))] + cols)

    try:
        coefs, *_ = np.linalg.lstsq(X, log_y, rcond=None)
    except Exception:
        return []

    # Check fit
    try:
        y_pred = np.exp(X @ coefs)
    except Exception:
        return []
    y_actual = np.array([y for _, y in data])
    err = float(np.mean(np.abs(y_pred - y_actual) / (np.abs(y_actual) + 1e-9)))
    if err > tol:
        return []

    # Reconstruct expression
    try:
        log_A = float(coefs[0])
        A = sp.Float(float(np.exp(log_A)))
        expr = A
        for v, k in zip(var_names, coefs[1:]):
            expr = expr * sp.Symbol(v) ** sp.Float(float(k))
        expr = sp.simplify(expr)
    except Exception:
        return []

    return [{"expr": expr, "fit_error": err, "method": "power_law"}]

