"""Minimum Description Length scoring.

Two-part code:
  L(model)  = bits to describe the expression (srepr canonical form)
  L(data|m) = bits to describe residuals under the model
Total L = L(model) + L(data|m). Lower is better.

Reject if max_residual / scale > tol, unless a single scalar fit
brings it inside tol (in which case report the rescaled expression).
"""
from __future__ import annotations
import math
import sympy as sp

BITS_PER_CHAR = 4.0


def _scale(y_vals: list[float]) -> float:
    n = len(y_vals)
    s = sorted(y_vals)
    med = s[n // 2]
    mad = sorted(abs(v - med) for v in y_vals)[n // 2]
    sc = 1.4826 * mad
    if sc < 1e-12:
        sc = (max(y_vals) - min(y_vals)) or 1.0
    return sc


def _residuals(expr: sp.Expr, data, scalar: float = 1.0) -> list[float] | None:
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


def _canonical(expr: sp.Expr) -> str:
    return sp.srepr(sp.simplify(expr))


def mdl_score(
    expr: sp.Expr,
    data: list[tuple[dict[str, float], float]],
    residual_tol: float = 0.02,
) -> dict:
    if not data:
        return {"bits": float("inf"), "residual": float("inf"),
                "size": 0, "ok": False}

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
        num = den = 0.0
        ok = True
        for inputs, y_true in data:
            try:
                v = float(expr.subs({sp.Symbol(k): x for k, x in inputs.items()}))
            except (TypeError, ValueError):
                ok = False
                break
            if not math.isfinite(v):
                ok = False
                break
            num += v * y_true
            den += v * v
        if not ok or den < 1e-12:
            return {"bits": float("inf"), "residual": max_res / scale,
                    "size": 0, "ok": False}
        c = num / den
        res2 = _residuals(expr, data, scalar=c)
        if res2 is None:
            return {"bits": float("inf"), "residual": max_res / scale,
                    "size": 0, "ok": False}
        max_res2 = max(res2)
        if max_res2 / scale > residual_tol:
            return {"bits": float("inf"), "residual": max_res2 / scale,
                    "size": 0, "ok": False}
        expr_used = sp.Float(round(c, 6)) * sp.simplify(expr)
        max_res = max_res2
        res = res2
        scalar = c

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
