"""Template-based symbolic regression.

Complements tree search: fits a fixed library of parametric forms by
least-squares, then ranks survivors by MDL via purediscover.mdl.

Why this exists:
    Tree search cannot always construct a function even when the
    evaluator knows it.  Templates reach forms the tree builder misses.
    Every template here is a physically common law family.

Usage:
    from purediscover_templates import search_templates
    results = search_templates(data, var_names=['x'])
    # -> [{'name': 'cos(a*x+b)+c', 'expr': 'cos(x)', 'mre': 8e-16, ...}]
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import sympy as sp
from scipy.optimize import curve_fit

from purediscover.mdl import mdl_score


# Each template: (name, p0, bounds, func(x, *p) -> y, sympy_builder(p) -> expr)

def _make(name, p0, bounds, func, sympy_builder):
    return dict(name=name, p0=p0, bounds=bounds, func=func, sympy=sympy_builder)


TEMPLATES = [
    _make(
        name="linear",
        p0=[1.0, 0.0], bounds=([-np.inf]*2, [np.inf]*2),
        func=lambda x, a, b: a*x + b,
        sympy_builder=lambda p: p[0]*sp.Symbol("x") + p[1],
    ),
    _make(
        name="quadratic",
        p0=[1.0, 0.0, 0.0],
        bounds=([-np.inf]*3, [np.inf]*3),
        func=lambda x, a, b, c: a*x**2 + b*x + c,
        sympy_builder=lambda p: p[0]*sp.Symbol("x")**2 + p[1]*sp.Symbol("x") + p[2],
    ),
    _make(
        name="power_law",
        p0=[1.0, 1.0, 0.0],
        bounds=([-np.inf, 0.01, -np.inf], [np.inf, 10.0, np.inf]),
        func=lambda x, a, b, c: a*np.power(np.abs(x), b) + c,
        sympy_builder=lambda p: p[0]*sp.Abs(sp.Symbol("x"))**p[1] + p[2],
    ),
    _make(
        name="exp(a*x)+c",
        p0=[1.0, 1.0, 0.0],
        bounds=([-np.inf, -10.0, -np.inf], [np.inf, 10.0, np.inf]),
        func=lambda x, a, b, c: a*np.exp(np.clip(b*x, -50, 50)) + c,
        sympy_builder=lambda p: p[0]*sp.exp(p[1]*sp.Symbol("x")) + p[2],
    ),
    _make(
        name="exp(-a*x^2)+c",
        p0=[1.0, 1.0, 0.0],
        bounds=([-np.inf, 0.001, -np.inf], [np.inf, 20.0, np.inf]),
        func=lambda x, a, b, c: a*np.exp(-b*x**2) + c,
        sympy_builder=lambda p: p[0]*sp.exp(-p[1]*sp.Symbol("x")**2) + p[2],
    ),
    _make(
        name="cos(a*x+b)+c",
        p0=[1.0, 1.0, 0.0, 0.0],
        bounds=([-np.inf, 0.001, -2*np.pi, -np.inf],
                [np.inf, 20.0, 2*np.pi, np.inf]),
        func=lambda x, a, b, c, d: a*np.cos(b*x + c) + d,
        sympy_builder=lambda p: (p[0]*sp.cos(p[1]*sp.Symbol("x") + p[2]) + p[3]),
    ),
    _make(
        name="sin(a*x+b)+c",
        p0=[1.0, 1.0, 0.0, 0.0],
        bounds=([-np.inf, 0.001, -2*np.pi, -np.inf],
                [np.inf, 20.0, 2*np.pi, np.inf]),
        func=lambda x, a, b, c, d: a*np.sin(b*x + c) + d,
        sympy_builder=lambda p: (p[0]*sp.sin(p[1]*sp.Symbol("x") + p[2]) + p[3]),
    ),
    _make(
        name="damped_cos",
        p0=[1.0, 1.0, 0.0, 0.1],
        bounds=([-np.inf, 0.001, -2*np.pi, 0.0],
                [np.inf, 20.0, 2*np.pi, 10.0]),
        func=lambda x, a, b, c, d: a*np.cos(b*x + c)*np.exp(-d*x),
        sympy_builder=lambda p: (p[0]*sp.cos(p[1]*sp.Symbol("x") + p[2])
                                 * sp.exp(-p[3]*sp.Symbol("x"))),
    ),
    _make(
        name="log(a*x)+c",
        p0=[1.0, 1.0, 0.0],
        bounds=([-np.inf, 0.001, -np.inf], [np.inf, 100.0, np.inf]),
        func=lambda x, a, b, c: a*np.log(b*np.abs(x) + 1e-30) + c,
        sympy_builder=lambda p: p[0]*sp.log(p[1]*sp.Abs(sp.Symbol("x"))) + p[2],
    ),
    _make(
        name="inverse",
        p0=[1.0, 1.0, 0.0],
        bounds=([-np.inf, 1e-6, -np.inf], [np.inf, 100.0, np.inf]),
        func=lambda x, a, b, c: a/(b + x) + c,
        sympy_builder=lambda p: p[0]/(p[1] + sp.Symbol("x")) + p[2],
    ),
    _make(
        name="sinc",
        p0=[1.0, 1.0, 0.0],
        bounds=([-np.inf, 0.001, -np.inf], [np.inf, 20.0, np.inf]),
        func=lambda x, a, b, c: a*np.sinc(b*x/np.pi) + c,
        sympy_builder=lambda p: (p[0]*sp.sin(p[1]*sp.Symbol("x"))
                                 / (p[1]*sp.Symbol("x") + 1e-30) + p[2]),
    ),
]


@dataclass
class TemplateResult:
    name: str
    expr_str: str
    params: list
    mre: float
    rms: float
    n_params: int
    mdl_bits: float = float("inf")
    ok: bool = False
    raw: dict = field(default_factory=dict)


def _mre(yp, ys):
    return float(np.mean(np.abs(yp - ys) / (np.abs(ys) + 1e-12)))


def _rms(yp, ys):
    return float(np.sqrt(np.mean((yp - ys) ** 2)))


def search_templates(
    data,
    var_names: list[str],
    *,
    target_names: list[str] | None = None,
    max_mre: float = 0.05,
    n_multistart: int = 4,
    seed: int = 0,
) -> list[TemplateResult]:
    """Fit every template by curve_fit; return MDL-ranked survivors.

    data: list[(dict(var->x), y)] or similar.  If var_names has one entry,
    this is simple 1D regression.  Multi-var is not yet supported by templates.
    """
    if len(var_names) != 1:
        raise ValueError(
            f"template_search currently supports 1 var, got {len(var_names)}"
        )
    xname = var_names[0]
    xs = np.array([d[0][xname] for d in data], dtype=float)
    ys = np.array([d[1] for d in data], dtype=float)
    x_sym = sp.Symbol("x")
    rng = np.random.default_rng(seed)

    results: list[TemplateResult] = []

    for tmpl in TEMPLATES:
        best_popt = None
        best_mre = float("inf")
        for _ in range(n_multistart):
            p0 = list(tmpl["p0"])
            if _ > 0:
                jitter = rng.normal(0.0, 0.3, size=len(p0))
                p0 = [float(np.clip(v + j, lo, hi))
                      for v, j, lo, hi in zip(p0, jitter,
                                              tmpl["bounds"][0],
                                              tmpl["bounds"][1])]
            try:
                popt, _ = curve_fit(
                    tmpl["func"], xs, ys, p0=p0,
                    bounds=tmpl["bounds"], maxfev=20000,
                )
            except Exception:
                continue
            yp = tmpl["func"](xs, *popt)
            if not np.all(np.isfinite(yp)):
                continue
            m = _mre(yp, ys)
            if m < best_mre:
                best_mre, best_popt = m, popt

        if best_popt is None:
            continue

        yp = tmpl["func"](xs, *best_popt)
        mre = _mre(yp, ys)
        rms = _rms(yp, ys)

        # build concrete sympy expr with floats substituted
        try:
            expr = tmpl["sympy"]([float(p) for p in best_popt])
            expr = expr.subs(x_sym, sp.Symbol(xname))
            expr_str = str(expr)
        except Exception:
            expr = None
            expr_str = "?"

        # MDL via purediscover
        mdl_bits = float("inf")
        ok = False
        if expr is not None and mre < max_mre:
            try:
                s = mdl_score(expr, data, residual_tol=max(mre * 3.0, 0.01))
                if isinstance(s, dict):
                    mdl_bits = float(s.get("bits", float("inf")))
                    ok = bool(s.get("ok", False))
            except Exception:
                pass

        results.append(TemplateResult(
            name=tmpl["name"],
            expr_str=expr_str,
            params=[float(p) for p in best_popt],
            mre=mre, rms=rms, n_params=len(best_popt),
            mdl_bits=mdl_bits, ok=ok,
            raw={"sympy_expr": expr},
        ))

    # rank by MDL (finite first), then by MRE
    def key(r):
        return (0 if math.isfinite(r.mdl_bits) else 1, r.mdl_bits, r.mre)

    results.sort(key=key)
    return results
