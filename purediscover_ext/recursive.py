"""Recursive residual discovery: decompose multi-component signals.

Strategy:
  1. Fit best candidate L1 from both arms (loose tolerance to bootstrap).
  2. Residuals r = y - L1(X).  If var(r) >> noise floor, try to fit a second
     law via two paths:
       additive:        L2_add = discover(r),    compose L1 + L2_add
       multiplicative:  L2_mul = discover(y/L1), compose L1 * L2_mul
  3. Pick the composition with lower MAE, recurse on the winning residual
     stream, up to max_depth.

Handles:
  - damped Rabi:      cos(w*t) * exp(-g*t)     -> multiplicative path
  - Keeling:          trend(t) + seasonal(t)   -> additive path
  - single-frequency: cos(4*J*t)               -> stops at depth 1
"""
from __future__ import annotations

import math

import numpy as np
import sympy as sp

from .tree_search_ext import tree_search_ext


def _run_template_arm(data, var_names):
    from purediscover_templates import (
        search_templates, search_templates_2d, search_templates_nd,
    )
    n = len(var_names)
    if n == 1:
        return search_templates(data, var_names, max_mre=0.50)
    if n == 2:
        return search_templates_2d(data, var_names, max_mre=0.50)
    return search_templates_nd(data, var_names, max_mre=0.50)


def _best_candidate(data, var_names, *, time_budget=25.0):
    pool = []
    try:
        templ = _run_template_arm(data, var_names)
        for t in (templ or []):
            bits = float(getattr(t, "mdl_bits", float("inf")))
            # TemplateResult (1D) has .mre; TemplateResult2D/ND have .mae.
            # Use whichever exists to avoid silently skipping all 1D templates.
            mae_raw = getattr(t, "mae", None)
            if mae_raw is None:
                mae_raw = getattr(t, "mre", float("inf"))
            mae = float(mae_raw)
            if not math.isfinite(mae):
                continue
            # If mdl_score rejected this candidate (bits=inf), assign a
            # fallback rank so it can still enter the pool.  Small MAE
            # candidates deserve consideration even if their residual
            # profile has outliers that MDL penalizes.
            if not math.isfinite(bits):
                if mae < 0.20:   # only rescue genuinely good fits
                    bits = 1.0e9 + mae * 1.0e6
                else:
                    continue
            raw = getattr(t, "raw", {}) or {}
            pool.append({
                "source": "templ",
                "expr": t.expr_str,
                "mae": mae,
                "bits": bits,
                "sympy_expr": raw.get("sympy_expr"),
            })
    except Exception:
        pass
    try:
        sparse = tree_search_ext(
            data, var_names,
            max_features=3, degree=2, include_trig=True,
            tol=0.02, time_budget=time_budget,
        )
        for s in (sparse or []):
            pool.append({
                "source": "sparse",
                "expr": s["expr"],
                "mae": float(s["mre"]),
                "bits": float(s["bits"]),
                "sympy_expr": s.get("sympy_expr"),
            })
    except Exception:
        pass
    if not pool:
        return None
    # Two-stage selection: keep candidates within 2x of best MAE, then pick
    # lowest MDL among those.  Pure MDL can select a simple wrong-shape fit
    # (e.g. 2-param cos with 5% MAE) over a more complex right-shape fit
    # (4-param damped cos with 1% MAE).  The near-optimal gate prevents this
    # while still honoring MDL as tiebreaker.
    min_mae = min(p["mae"] for p in pool)
    near_optimal = [p for p in pool
                    if p["mae"] <= 2.0 * min_mae + 1e-9]
    near_optimal.sort(key=lambda p: (p["bits"], p["mae"]))
    return near_optimal[0]


def _predict(expr, data, var_names):
    if expr is None:
        return None
    x_syms = [sp.Symbol(v) for v in var_names]
    xs = [np.array([d[0][v] for d in data], dtype=float) for v in var_names]
    ys = np.array([d[1] for d in data], dtype=float)
    try:
        f = sp.lambdify(x_syms, expr, modules=["numpy"])
        yp = f(*xs)
        if np.isscalar(yp):
            yp = np.full(len(ys), float(yp))
        yp = np.asarray(yp, dtype=float)
        if not np.all(np.isfinite(yp)):
            return None
        return yp, ys
    except Exception:
        return None


def _composed_mae(l1_expr, l2_expr, data, var_names, mode):
    r1 = _predict(l1_expr, data, var_names)
    r2 = _predict(l2_expr, data, var_names)
    if r1 is None or r2 is None:
        return float("inf")
    y1, ys = r1
    y2, _ = r2
    pred = y1 + y2 if mode == "additive" else y1 * y2
    if not np.all(np.isfinite(pred)):
        return float("inf")
    return float(np.mean(np.abs(pred - ys)))


def _expr_of(cand):
    e = cand.get("sympy_expr")
    if e is not None:
        return e
    try:
        return sp.sympify(cand["expr"])
    except Exception:
        return None


def discover_recursive(
    data,
    var_names,
    *,
    max_depth=2,
    noise_floor_factor=0.02,
    residual_var_factor=3.0,
    time_budget=25.0,
    verbose=False,
):
    """Return dict with laws, composed_expr, composition_mode, final_mae, depth."""
    ys = np.array([d[1] for d in data], dtype=float)
    noise_floor = noise_floor_factor * float(np.max(np.abs(ys)) + 1e-12)

    laws = []
    current_data = list(data)
    composition_mode = "single"
    final_composed_expr = None
    final_composed_mae = None

    for depth in range(max_depth):
        cand = _best_candidate(current_data, var_names, time_budget=time_budget)
        if cand is None:
            break

        laws.append(cand)
        if verbose:
            print(f"[depth {depth}] {cand['source']:6s} "
                  f"mae={cand['mae']:.3e} bits={cand['bits']:.1f}")
            print(f"             {cand['expr'][:100]}")

        if depth == 0:
            final_composed_expr = cand["expr"]
            final_composed_mae = cand["mae"]

        if depth >= max_depth - 1:
            break

        pred = _predict(cand["sympy_expr"], current_data, var_names)
        if pred is None:
            break
        yp, ys_cur = pred
        residuals = ys_cur - yp
        var_res = float(np.var(residuals))
        if var_res < (residual_var_factor * noise_floor) ** 2:
            if verbose:
                print(f"[depth {depth}] residuals flat "
                      f"(var={var_res:.3e}) -> stop")
            break

        # --- additive path ---
        res_data = [({v: current_data[i][0][v] for v in var_names},
                     float(residuals[i])) for i in range(len(residuals))]
        L2_add = _best_candidate(res_data, var_names, time_budget=time_budget)
        mae_add = (float("inf")
                   if L2_add is None
                   else _composed_mae(cand["sympy_expr"],
                                      L2_add["sympy_expr"],
                                      current_data, var_names, "additive"))

        # --- multiplicative path ---
        mae_mul = float("inf")
        L2_mul = None
        ratio_data = None
        if cand["sympy_expr"] is not None:
            x_syms = [sp.Symbol(v) for v in var_names]
            xs = [np.array([d[0][v] for d in current_data], dtype=float)
                  for v in var_names]
            try:
                f = sp.lambdify(x_syms, cand["sympy_expr"], modules=["numpy"])
                yp_div = f(*xs)
                if np.isscalar(yp_div):
                    yp_div = np.full(len(ys_cur), float(yp_div))
                yp_div = np.asarray(yp_div, dtype=float)
                mask = np.abs(yp_div) > 1e-3
                if mask.sum() >= max(5, len(ys_cur) // 2):
                    ratio_data = [
                        ({v: current_data[i][0][v] for v in var_names},
                         float(ys_cur[i] / yp_div[i]))
                        for i in np.where(mask)[0]
                    ]
                    L2_mul = _best_candidate(ratio_data, var_names,
                                              time_budget=time_budget)
                    if L2_mul is not None:
                        mae_mul = _composed_mae(cand["sympy_expr"],
                                                 L2_mul["sympy_expr"],
                                                 current_data, var_names,
                                                 "multiplicative")
            except Exception:
                pass

        if mae_add < mae_mul and L2_add is not None:
            mode, L2, c_mae = "additive", L2_add, mae_add
            current_data = res_data
        elif mae_mul < mae_add and L2_mul is not None:
            mode, L2, c_mae = "multiplicative", L2_mul, mae_mul
            current_data = ratio_data
        else:
            break

        e1 = _expr_of(cand)
        e2 = _expr_of(L2)
        if e1 is None or e2 is None:
            break
        composed = (e1 + e2) if mode == "additive" else (e1 * e2)
        final_composed_expr = str(composed)
        final_composed_mae = c_mae
        composition_mode = mode

        if verbose:
            print(f"[depth {depth}] {mode} composite: mae={c_mae:.3e}")
            print(f"             {final_composed_expr[:120]}")

    # Robust fallback: if the stored MAE is inf/None, re-evaluate the
    # base candidate directly.  Stored MAE can be stale from a failed
    # composition path even when the base law itself is fine.
    if laws and (final_composed_mae is None
                 or not math.isfinite(final_composed_mae)):
        e0 = _expr_of(laws[0])
        if e0 is not None:
            pred = _predict(e0, data, var_names)
            if pred is not None:
                yp0, ys0 = pred
                final_composed_mae = float(np.mean(np.abs(yp0 - ys0)))
            else:
                final_composed_mae = laws[0]["mae"]
        else:
            final_composed_mae = laws[0]["mae"]
        final_composed_expr = laws[0]["expr"]
        composition_mode = "single"

    return {
        "laws": laws,
        "composed_expr": final_composed_expr,
        "composition_mode": composition_mode,
        "final_mae": final_composed_mae,
        "depth": len(laws),
    }
