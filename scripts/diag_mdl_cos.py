#!/usr/bin/env python3
"""Confirm that mdl_score cannot evaluate cos, then brute-force with sympy+lambdify."""
import inspect
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import sympy as sp

from purediscover.mdl import mdl_score
from purediscover.compose import brute_search


def main() -> int:
    pts = json.loads((_ROOT / "results/closed_loop/state.json").read_text())["data"]
    data = [({"Bx": p["Bx"]}, p["y"]) for p in pts]
    Bx = sp.Symbol("Bx")

    print("=== 1. mdl_score on candidate expressions ===")
    for expr in [sp.exp(-sp.Rational(3, 5) * Bx**2),
                 sp.cos(Bx),
                 sp.sin(Bx),
                 sp.cos(sp.Symbol("a") * Bx)]:
        try:
            r = mdl_score(expr, data, residual_tol=0.05)
            print(f"  {str(expr):40s} -> {r}")
        except Exception as e:
            print(f"  {str(expr):40s} -> EXCEPTION: {type(e).__name__}: {e}")

    print()
    print("=== 2. brute_search evaluator source ===")
    try:
        src = inspect.getsource(brute_search)
        for i, line in enumerate(src.split("\n")):
            low = line.lower()
            if any(k in low for k in ("lambdify", "eval", "modules",
                                       "trans", "fn_map", "funcs",
                                       "np.", "math.", "sin", "cos")):
                print(f"  L{i:03d}: {line.rstrip()}")
    except Exception as e:
        print(f"  source unavailable: {e}")

    print()
    print("=== 3. mdl_score source (evaluator line) ===")
    try:
        src = inspect.getsource(mdl_score)
        for i, line in enumerate(src.split("\n")):
            low = line.lower()
            if any(k in low for k in ("lambdify", "eval", "modules",
                                       "subs", "pycode", "resid", "np.")):
                print(f"  L{i:03d}: {line.rstrip()}")
    except Exception as e:
        print(f"  source unavailable: {e}")

    print()
    print("=== 4. self-contained brute-force with lambdify ===")
    # templates to try — all evaluated with numpy modules (knows cos, sin)
    xs = np.array([d[0]["Bx"] for d in data], dtype=float)
    ys = np.array([d[1] for d in data], dtype=float)

    def lambdify_f(expr, symbols):
        f = sp.lambdify(symbols, expr, modules=["numpy"])
        return f

    templates = [
        ("cos(x)",          [Bx],            lambda s: sp.cos(s[0])),
        ("sin(x)",          [Bx],            lambda s: sp.sin(s[0])),
        ("cos(a*x)",        [Bx, sp.Symbol("a")],  None),
        ("1 - x**2/2",      [Bx],            lambda s: 1 - s[0]**2 / 2),
        ("1 - x**2/2 + x**4/24", [Bx],       lambda s: 1 - s[0]**2/2 + s[0]**4/24),
        ("exp(-a*x**2)",    [Bx, sp.Symbol("a")],  None),
    ]

    best = None
    # 1-symbol templates: direct lambdify
    for name, syms, builder in templates:
        if builder is None:
            continue
        try:
            expr = builder(syms)
            f = sp.lambdify(syms, expr, modules=["numpy"])
            yp = f(xs)
            yp = np.broadcast_to(yp, xs.shape).astype(float)
            mre = float(np.mean(np.abs(yp - ys) / (np.abs(ys) + 1e-12)))
            print(f"  {name:28s}  MRE={mre:.3e}  expr={expr}")
            if best is None or mre < best[0]:
                best = (mre, expr, name)
        except Exception as e:
            print(f"  {name:28s}  FAILED: {e}")

    # parametric templates: fit with scipy curve_fit
    try:
        from scipy.optimize import curve_fit
        # cos(a*x + b)
        def f_cos(x, a, b): return np.cos(a * x + b)
        popt, _ = curve_fit(f_cos, xs, ys, p0=[1.0, 0.0], maxfev=20000)
        yp = f_cos(xs, *popt)
        mre = float(np.mean(np.abs(yp - ys) / (np.abs(ys) + 1e-12)))
        print(f"  {'cos(a*x+b) (fitted)':28s}  MRE={mre:.3e}  "
              f"a={popt[0]:.6f}  b={popt[1]:.6f}")
        if mre < best[0]:
            best = (mre, f"cos({popt[0]:.6f}*Bx + {popt[1]:.6f})",
                    "cos(a*x+b)")
        # exp(-a*x**2)
        def f_exp(x, a): return np.exp(-a * x**2)
        popt, _ = curve_fit(f_exp, xs, ys, p0=[0.5], maxfev=20000)
        yp = f_exp(xs, *popt)
        mre = float(np.mean(np.abs(yp - ys) / (np.abs(ys) + 1e-12)))
        print(f"  {'exp(-a*x^2) (fitted)':28s}  MRE={mre:.3e}  a={popt[0]:.6f}")
    except Exception as e:
        print(f"  curve_fit failed: {e}")

    print()
    print("=== 5. best reconstruction ===")
    if best:
        print(f"  name: {best[2]}")
        print(f"  expr: {best[1]}")
        print(f"  MRE : {best[0]:.3e}")
    print()
    print("HINT: analytic answer for <Z>(Bx) at t=0.5 on |0>, H=Bx*X is cos(Bx).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
