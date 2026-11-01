"""2D template search. MAE primary, MRE (floored) secondary."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import sympy as sp
from scipy.optimize import curve_fit

from purediscover.mdl import mdl_score


def _mk(name, p0, lo, hi, func, sym_builder):
    return dict(name=name, p0=p0, bounds=(lo, hi),
                func=func, sympy=sym_builder)


TEMPLATES_2D = [
    _mk("linear", [1.0, 1.0, 0.0], [-np.inf]*3, [np.inf]*3,
        lambda xy, a, b, c: a*xy[0] + b*xy[1] + c,
        lambda p, x, y: p[0]*x + p[1]*y + p[2]),
    _mk("bilinear", [1.0, 0.0, 0.0, 0.0], [-np.inf]*4, [np.inf]*4,
        lambda xy, a, b, c, d: a*xy[0]*xy[1] + b*xy[0] + c*xy[1] + d,
        lambda p, x, y: p[0]*x*y + p[1]*x + p[2]*y + p[3]),
    _mk("cos(a*x*y)", [1.0, 1.0], [-np.inf, 0.001], [np.inf, 30.0],
        lambda xy, a, b: a*np.cos(b*xy[0]*xy[1]),
        lambda p, x, y: p[0]*sp.cos(p[1]*x*y)),
    _mk("cos(a*x*y+b)", [1.0, 1.0, 0.0],
        [-np.inf, 0.001, -2*np.pi], [np.inf, 30.0, 2*np.pi],
        lambda xy, a, b, c: a*np.cos(b*xy[0]*xy[1] + c),
        lambda p, x, y: p[0]*sp.cos(p[1]*x*y + p[2])),
    _mk("cos(a*x+b*y+c)", [1.0, 1.0, 1.0, 0.0],
        [-np.inf, 0.001, 0.001, -2*np.pi], [np.inf, 30.0, 30.0, 2*np.pi],
        lambda xy, a, b, c, d: a*np.cos(b*xy[0] + c*xy[1] + d),
        lambda p, x, y: p[0]*sp.cos(p[1]*x + p[2]*y + p[3])),
    _mk("sin(a*x*y)", [1.0, 1.0], [-np.inf, 0.001], [np.inf, 30.0],
        lambda xy, a, b: a*np.sin(b*xy[0]*xy[1]),
        lambda p, x, y: p[0]*sp.sin(p[1]*x*y)),
    _mk("sin(a*x*y+b)", [1.0, 1.0, 0.0],
        [-np.inf, 0.001, -2*np.pi], [np.inf, 30.0, 2*np.pi],
        lambda xy, a, b, c: a*np.sin(b*xy[0]*xy[1] + c),
        lambda p, x, y: p[0]*sp.sin(p[1]*x*y + p[2])),
    _mk("exp(-a*x*y+b)", [1.0, 1.0, 0.0],
        [-np.inf, 0.001, -10.0], [np.inf, 30.0, 10.0],
        lambda xy, a, b, c: a*np.exp(-b*xy[0]*xy[1] + c),
        lambda p, x, y: p[0]*sp.exp(-p[1]*x*y + p[2])),
    _mk("exp(-a*x-b*y+c)", [1.0, 1.0, 1.0, 0.0],
        [-np.inf, 0.001, 0.001, -10.0], [np.inf, 30.0, 30.0, 10.0],
        lambda xy, a, b, c, d: a*np.exp(-b*xy[0] - c*xy[1] + d),
        lambda p, x, y: p[0]*sp.exp(-p[1]*x - p[2]*y + p[3])),
    _mk("power_product", [1.0, 1.0, 1.0, 0.0],
        [-np.inf, -10, -10, -np.inf], [np.inf, 10, 10, np.inf],
        lambda xy, a, p_, q, c: a
            * np.power(np.abs(xy[0]) + 1e-12, p_)
            * np.power(np.abs(xy[1]) + 1e-12, q) + c,
        lambda p, x, y: p[0]*sp.Pow(sp.Abs(x)+1e-12, p[1])
                          * sp.Pow(sp.Abs(y)+1e-12, p[2]) + p[3]),
]


@dataclass
class TemplateResult2D:
    name: str
    expr_str: str
    params: list
    mae: float
    mre: float
    rms: float
    n_params: int
    mdl_bits: float = float("inf")
    ok: bool = False
    raw: dict = field(default_factory=dict)
    mre_legacy: float = 0.0


def _mae(yp, ys):
    return float(np.mean(np.abs(yp - ys)))


def _mre_floored(yp, ys):
    y_floor = max(0.05 * float(np.max(np.abs(ys))), 1e-9)
    return float(np.mean(np.abs(yp - ys) / np.maximum(np.abs(ys), y_floor)))


def _mre_legacy(yp, ys):
    return float(np.mean(np.abs(yp - ys) / (np.abs(ys) + 1e-12)))


def _rms(yp, ys):
    return float(np.sqrt(np.mean((yp - ys) ** 2)))


def search_templates_2d(data, var_names, *, max_mae=None, max_mre=0.10,
                       n_multistart=5, seed=0, verbose=False):
    if len(var_names) != 2:
        raise ValueError(f"needs 2 vars, got {var_names}")
    xn, yn = var_names
    xs = np.array([d[0][xn] for d in data], dtype=float)
    ys_var = np.array([d[0][yn] for d in data], dtype=float)
    zs = np.array([d[1] for d in data], dtype=float)
    xdata = np.vstack([xs, ys_var])

    rng = np.random.default_rng(seed)
    results: list[TemplateResult2D] = []
    n_ok_fit = 0

    for tmpl in TEMPLATES_2D:
        best_popt, best_mae = None, float("inf")
        for k in range(n_multistart):
            p0 = list(tmpl["p0"])
            if k:
                j = rng.normal(0.0, 0.3, len(p0))
                p0 = [float(np.clip(v + dj, lo, hi))
                      for v, dj, lo, hi in zip(p0, j, tmpl["bounds"][0],
                                               tmpl["bounds"][1])]
            try:
                popt, _ = curve_fit(tmpl["func"], xdata, zs, p0=p0,
                                    bounds=tmpl["bounds"], maxfev=20000)
            except Exception as e:
                if verbose:
                    print(f"  [{tmpl['name']}] fit failed: {e}")
                continue
            yp = tmpl["func"](xdata, *popt)
            if not np.all(np.isfinite(yp)):
                continue
            m = _mae(yp, zs)
            if m < best_mae:
                best_mae, best_popt = m, popt
        if best_popt is None:
            continue
        n_ok_fit += 1

        yp = tmpl["func"](xdata, *best_popt)
        mae = _mae(yp, zs)
        mre = _mre_floored(yp, zs)
        mre_old = _mre_legacy(yp, zs)
        try:
            expr = tmpl["sympy"]([float(p) for p in best_popt],
                                 sp.Symbol(xn), sp.Symbol(yn))
            expr_str = str(expr)
        except Exception:
            expr, expr_str = None, "?"

        bits, ok = float("inf"), False
        passes = mre < max_mre
        if max_mae is not None and mae > max_mae:
            passes = False
        if expr is not None and passes:
            try:
                s = mdl_score(expr, data, residual_tol=max(mre * 6.0, 0.03))
                if isinstance(s, dict):
                    bits = float(s.get("bits", float("inf")))
                    ok = bool(s.get("ok", False))
            except Exception:
                pass

        results.append(TemplateResult2D(
            name=tmpl["name"], expr_str=expr_str,
            params=[float(p) for p in best_popt],
            mae=mae, mre=mre, rms=_rms(yp, zs), mre_legacy=mre_old,
            n_params=len(best_popt), mdl_bits=bits, ok=ok,
            raw={"sympy_expr": expr},
        ))

    if verbose:
        print(f"  fitted {n_ok_fit}/{len(TEMPLATES_2D)} templates")

    results.sort(key=lambda r: (0 if math.isfinite(r.mdl_bits) else 1,
                                r.mdl_bits, r.mae))
    return results
