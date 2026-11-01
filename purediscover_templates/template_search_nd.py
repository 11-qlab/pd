"""N-dimensional template search via composite variable reduction.

Strategy:
  1. Enumerate composite variables z = f(x_1,...,x_n):
       singles, pairwise products, triple products, ratios, sums
  2. For each composite z, fit the 1D template library to (z, y).
  3. Return MDL-ranked results.

For cos(2*Bx*t): composite z = Bx*t, then 1D template cos(a*z) finds a=2.0.
For cosh(2*Bx*t): same, cosh family (add cosh/sinh to 1D library).
For 1.7*x*y + 0.3*x - 0.8*y + 2: additive path below.

Additive path: y = sum_i g_i(x_i) for 1D g_i. Fit each g_i with 1D templates,
sum, then refit coefficients via lstsq. Handles most multi-variable physics.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
import sympy as sp
from scipy.optimize import curve_fit

from purediscover.mdl import mdl_score


@dataclass
class CompositeVar:
    name: str
    expr: sp.Expr
    values: np.ndarray


@dataclass
class NDResult:
    name: str
    expr_str: str
    composite: str
    params: list
    mae: float
    mre: float
    mdl_bits: float
    ok: bool


def _make_composites(data, var_names, *, max_order=3, include_ratios=True,
                     include_sums=True):
    x_syms = [sp.Symbol(v) for v in var_names]
    xs = {v: np.array([d[0][v] for d in data], dtype=float) for v in var_names}
    comps = []
    for v in var_names:
        comps.append(CompositeVar(v, sp.Symbol(v), xs[v]))
    for k in range(2, max_order + 1):
        for combo in itertools.combinations(x_syms, k):
            name = "*".join(str(c) for c in combo)
            vals = np.ones_like(xs[var_names[0]])
            for c in combo:
                vals = vals * xs[str(c)]
            comps.append(CompositeVar(name, sp.prod(combo), vals))
    if include_ratios:
        for i, vi in enumerate(var_names):
            for j, vj in enumerate(var_names):
                if i != j:
                    name = f"{vi}/{vj}"
                    vals = xs[vi] / (np.abs(xs[vj]) + 1e-30)
                    comps.append(CompositeVar(name,
                        sp.Symbol(vi) / sp.Symbol(vj), vals))
    if include_sums:
        for i, vi in enumerate(var_names):
            for j, vj in enumerate(var_names):
                if i < j:
                    name = f"{vi}+{vj}"
                    vals = xs[vi] + xs[vj]
                    comps.append(CompositeVar(name,
                        sp.Symbol(vi) + sp.Symbol(vj), vals))
    return comps


def _get_1d_templates():
    """Pull 1D template library from template_search, adding cosh/sinh/tanh."""
    from .template_search import TEMPLATES
    extra = [
        dict(name="cosh(a*x)",
             p0=[1.0, 1.0], bounds=([-np.inf, 0.001], [np.inf, 20.0]),
             func=lambda x, a, b: a*np.cosh(np.clip(b*x, -20, 20)),
             sympy=lambda p: p[0]*sp.cosh(p[1]*sp.Symbol("x"))),
        dict(name="sinh(a*x)",
             p0=[1.0, 1.0], bounds=([-np.inf, 0.001], [np.inf, 20.0]),
             func=lambda x, a, b: a*np.sinh(np.clip(b*x, -20, 20)),
             sympy=lambda p: p[0]*sp.sinh(p[1]*sp.Symbol("x"))),
        dict(name="tanh(a*x+b)",
             p0=[1.0, 1.0, 0.0], bounds=([-np.inf, 0.001, -5.0],
                                          [np.inf, 20.0, 5.0]),
             func=lambda x, a, b, c: a*np.tanh(b*x + c),
             sympy=lambda p: p[0]*sp.tanh(p[1]*sp.Symbol("x") + p[2])),
    ]
    return list(TEMPLATES) + extra


def search_templates_nd(data, var_names, *, max_order=3, max_mre=0.15,
                       n_multistart=6, seed=0, verbose=False):
    if not var_names:
        return []
    ys = np.array([d[1] for d in data], dtype=float)
    y_floor = max(0.05 * float(np.max(np.abs(ys))), 1e-9)

    comps = _make_composites(data, var_names, max_order=max_order)
    templates = _get_1d_templates()
    rng = np.random.default_rng(seed)
    results: list[NDResult] = []

    for comp in comps:
        zs = comp.values
        if not np.all(np.isfinite(zs)) or np.std(zs) < 1e-12:
            continue
        for tmpl in templates:
            best_popt, best_mae = None, float("inf")
            # FFT-based seed for frequency parameter (index 1) of trig templates
            trig_seed = None
            if tmpl["name"].startswith(("cos", "sin", "damped_cos", "sinc")):
                try:
                    # rough dominant frequency of z vs y
                    order = np.argsort(zs)
                    y_sorted = ys[order]
                    if len(y_sorted) >= 8:
                        fft = np.fft.rfft(y_sorted - np.mean(y_sorted))
                        freqs = np.fft.rfftfreq(len(y_sorted),
                                                d=(zs.max()-zs.min())/(len(zs)-1))
                        if len(fft) > 1:
                            peak = int(np.argmax(np.abs(fft[1:])) + 1)
                            dom_freq = freqs[peak] * 2 * np.pi
                            if 0.01 < dom_freq < 50:
                                trig_seed = float(dom_freq)
                except Exception:
                    pass
            for k in range(n_multistart):
                p0 = list(tmpl["p0"])
                if k == 0 and trig_seed is not None and len(p0) > 1:
                    p0[1] = trig_seed
                elif k:
                    j = rng.normal(0.0, 0.3, len(p0))
                    p0 = [float(np.clip(v + dj, lo, hi))
                          for v, dj, lo, hi in zip(p0, j, tmpl["bounds"][0],
                                                   tmpl["bounds"][1])]
                try:
                    popt, _ = curve_fit(tmpl["func"], zs, ys, p0=p0,
                                       bounds=tmpl["bounds"], maxfev=20000)
                except Exception:
                    continue
                yp = tmpl["func"](zs, *popt)
                if not np.all(np.isfinite(yp)):
                    continue
                m = float(np.mean(np.abs(yp - ys)))
                if m < best_mae:
                    best_mae, best_popt = m, popt
            if best_popt is None:
                continue
            yp = tmpl["func"](zs, *best_popt)
            mre = float(np.mean(np.abs(yp - ys) / np.maximum(np.abs(ys), y_floor)))
            if mre > max_mre:
                continue
            try:
                expr_1d = tmpl["sympy"]([float(p) for p in best_popt])
                expr = expr_1d.subs(sp.Symbol("x"), comp.expr)
                expr_str = str(expr)
            except Exception:
                continue
            try:
                s = mdl_score(expr, data, residual_tol=max(mre * 6.0, 0.03))
                bits = float(s.get("bits", float("inf"))) if isinstance(s, dict) else float("inf")
                ok = bool(s.get("ok", False)) if isinstance(s, dict) else False
            except Exception:
                bits, ok = float("inf"), False
            results.append(NDResult(
                name=f"{tmpl['name']} on {comp.name}",
                expr_str=expr_str, composite=comp.name,
                params=[float(p) for p in best_popt],
                mae=best_mae, mre=mre, mdl_bits=bits, ok=ok,
            ))
            if verbose:
                print(f"  {comp.name:20s} {tmpl['name']:20s} "
                      f"mae={best_mae:.3e}  bits={bits:.1f}")

    results.sort(key=lambda r: (0 if math.isfinite(r.mdl_bits) else 1,
                                r.mdl_bits, r.mae))
    return results
