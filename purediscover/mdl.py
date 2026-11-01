"""Minimum Description Length scoring.

Two formulas coexist:

  mdl_score (v1)     — legacy; over-weights structural bits. Kept so previously
                        reported bit counts remain reproducible.
  mdl_score_v2       — correct two-part code with Gaussian universal coding for
                        residuals, per-node structural cost, and a small
                        log2(n) parameter penalty.

v1 bias: data_bits = sum(log2(1 + r/scale)) grows ~linearly in r/scale, so a
100x fit improvement buys only ~1 bit for n=100, while one extra node costs
~40 bits. Effectively demands 3000x fit gains to justify one node.

v2: L = n_nodes*log2(|lib|) + n_const*log2(RANGE) + (n/2)*log2(2*pi*e*RSS/n).
Now the RSS term scales as (n/2)*log2, so 2x better RSS on n=100 buys 50 bits.
"""
from __future__ import annotations

import math
import sympy as sp

BITS_PER_CHAR = 4.0            # legacy (v1 only)
DEFAULT_LIBRARY_SIZE = 20      # ops + functions in a typical grammar
DEFAULT_CONST_RANGE = 200.0    # numbers assumed to lie in [-100, 100]
PERFECT_FIT_FLOOR = -30.0      # bits per residual when RSS ~ 0


def _scale(y_vals):
    n = len(y_vals)
    s = sorted(y_vals)
    med = s[n // 2]
    mad = sorted(abs(v - med) for v in y_vals)[n // 2]
    sc = 1.4826 * mad
    if sc < 1e-12:
        sc = (max(y_vals) - min(y_vals)) or 1.0
    return sc


def _residuals(expr, data, scalar=1.0):
    out = []
    for inputs, y_true in data:
        try:
            val = expr.subs({sp.Symbol(k): v for k, v in inputs.items()})
            y_pred = scalar * float(val)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(y_pred):
            return None
        out.append(abs(y_pred - y_true))
    return out


def _canonical(expr):
    return sp.srepr(sp.simplify(expr))


def _rescale(expr, data, scale, tol):
    """Shared rescale path for both scorers. Returns (expr, res, max_res, scalar) or None."""
    num = den = 0.0
    for inputs, y_true in data:
        try:
            v = float(expr.subs({sp.Symbol(k): x for k, x in inputs.items()}))
        except (TypeError, ValueError):
            return None
        if not math.isfinite(v):
            return None
        num += v * y_true
        den += v * v
    if den < 1e-12:
        return None
    c = num / den
    res2 = _residuals(expr, data, scalar=c)
    if res2 is None:
        return None
    max_res2 = max(res2)
    if max_res2 / scale > tol:
        return None
    return sp.Float(round(c, 6)) * sp.simplify(expr), res2, max_res2, c


# ================================================================ v1 (legacy)
def _mdl_score_v1(expr, data, residual_tol=0.02):
    if not data:
        return {"bits": float("inf"), "residual": float("inf"), "size": 0, "ok": False}

    y_vals = [y for _, y in data]
    scale = _scale(y_vals)
    res = _residuals(expr, data)
    if res is None:
        return {"bits": float("inf"), "residual": float("inf"), "size": 0, "ok": False}

    max_res = max(res)
    expr_used = expr
    scalar = 1.0

    if max_res / scale > residual_tol:
        out = _rescale(expr, data, scale, residual_tol)
        if out is None:
            return {"bits": float("inf"), "residual": max_res / scale,
                    "size": 0, "ok": False}
        expr_used, res, max_res, scalar = out

    residual_ratio = max_res / scale
    data_bits = sum(math.log2(1.0 + r / scale + 1e-12) for r in res)
    can = _canonical(expr_used)
    model_bits = len(can) * BITS_PER_CHAR

    return {
        "bits": model_bits + data_bits,
        "residual": residual_ratio,
        "size": len(can),
        "expr": expr_used,
        "scalar": scalar,
        "ok": True,
    }


# ================================================================ v2 (correct)
def _count_ast_nodes(expr):
    n = 1
    for arg in expr.args:
        n += _count_ast_nodes(arg)
    return n


def _count_free_params(expr):
    """Count non-trivial numeric constants. Excludes 0, ±1 (structural)."""
    count = 0
    for node in sp.preorder_traversal(expr):
        if not node.is_number:
            continue
        if isinstance(node, sp.Integer):
            if abs(int(node)) > 1:
                count += 1
        else:
            count += 1  # Rational, Float, ...
    return count


def mdl_score_v2(expr, data, residual_tol=0.02,
                 library_size=DEFAULT_LIBRARY_SIZE,
                 const_range=DEFAULT_CONST_RANGE):
    """Correct two-part MDL.

    L(expr)   = n_nodes * log2(library_size) + n_const * log2(const_range)
    L(data|m) = (n/2) * log2(2 * pi * e * RSS / n)

    Lower is better. Same return shape as mdl_score, plus 'components'.
    """
    if not data:
        return {"bits": float("inf"), "residual": float("inf"),
                "size": 0, "ok": False}

    n = len(data)
    y_vals = [y for _, y in data]
    scale = _scale(y_vals)
    res = _residuals(expr, data)
    if res is None:
        return {"bits": float("inf"), "residual": float("inf"),
                "size": 0, "ok": False}

    max_res = max(res)
    expr_used = expr
    scalar = 1.0

    if max_res / scale > residual_tol:
        out = _rescale(expr, data, scale, residual_tol)
        if out is None:
            return {"bits": float("inf"), "residual": max_res / scale,
                    "size": 0, "ok": False}
        expr_used, res, max_res, scalar = out

    residual_ratio = max_res / scale

    n_nodes = _count_ast_nodes(expr_used)
    n_const = _count_free_params(expr_used)

    L_struct = n_nodes * math.log2(library_size)
    L_const = n_const * math.log2(const_range)

    rss = sum(r * r for r in res)
    # Floor RSS at machine-precision scale of the data to keep L_data finite.
    rss_floor = n * (scale * 1e-10) ** 2
    rss_eff = max(rss, rss_floor)
    L_data = (n / 2.0) * math.log2(2.0 * math.pi * math.e * rss_eff / n)

    # Null baseline: predict the median everywhere.
    med = sorted(y_vals)[n // 2]
    rss_null = sum((v - med) ** 2 for v in y_vals)
    rss_null = max(rss_null, rss_floor)
    L_null = (n / 2.0) * math.log2(2.0 * math.pi * math.e * rss_null / n)

    # Report gain vs null (negative = compresses better than null).
    L_data_gain = L_data - L_null

    total = L_struct + L_const + L_data_gain

    return {
        "bits": total,
        "residual": residual_ratio,
        "size": n_nodes,
        "expr": expr_used,
        "scalar": scalar,
        "components": {"struct": L_struct, "const": L_const,
                       "data": L_data, "data_gain": L_data_gain},
        "ok": True,
    }


# ================================================================ public alias
import os
def mdl_score(expr, data, residual_tol=0.02):
    """Public entry point. Set PUREDISCOVER_MDL=v1|v2 to select scorer."""
    if os.environ.get("PUREDISCOVER_MDL", "v1") == "v2":
        return mdl_score_v2(expr, data, residual_tol=residual_tol)
    return _mdl_score_v1(expr, data, residual_tol=residual_tol)
