"""End-to-end symbolic regression orchestrator (rev 2).

Routes each task to the best tool:
  1D algebraic -> purediscover transform search
  multivariate -> PySR (genetic programming)
  implicit F=0 -> Sympy.solve + 1D
"""
from __future__ import annotations
import numpy as np
import sympy as sp
import traceback

from purediscover_ext.unknown_classical import discover_structure, TRANSFORMS

try:
    from pysr import PySRRegressor
    HAVE_PYSR = True
except ImportError:
    HAVE_PYSR = False


def _p90(y_pred, y_true):
    y_pred = np.asarray(y_pred, float)
    y_true = np.asarray(y_true, float)
    if not np.all(np.isfinite(y_pred)):
        return float("inf")
    rel = np.abs(y_pred - y_true) / np.maximum(np.abs(y_true), 1e-10)
    return float(np.percentile(rel, 90))


def _apply_1d(best, x):
    tf = TRANSFORMS[best["transform_x"]][0]
    ti = TRANSFORMS[best["transform_y"]][1]
    return ti(best["slope"] * tf(np.asarray(x, float)) + best["intercept"])


def _try_1d(x, y):
    r = discover_structure(
        [{"x": {"x": float(a)}, "y": float(b)} for a, b in zip(x, y)],
        ["x"], "y",
    )
    if r["best"] is None:
        return None
    p = _p90(_apply_1d(r["best"], x), y)
    if p < 0.05:
        return {"strategy": "1d_transform", "equation": r["expression"],
                "p90": p, "ok": True}
    return None


def _try_multivariate_pysr(X, y, timeout_sec=180, verbose=False):
    """PySR with API-compatible constructor. Handles PySR 0.x and 1.x+."""
    if not HAVE_PYSR:
        return {"strategy": "pysr", "error": "PySR not installed", "ok": False}

    # Build kwargs. Do NOT set deterministic/random_state —
    # PySR requires parallelism='serial' for those, which is 10x slower.
    kwargs = dict(
        niterations=20,
        binary_operators=["+", "*", "-", "/"],
        unary_operators=["square", "sqrt", "exp", "log"],
        maxsize=25,
        timeout_in_seconds=timeout_sec,
        progress=False,
        verbosity=0,
    )
    # Add `populations` only if the current PySR version supports it
    import inspect
    sig = inspect.signature(PySRRegressor.__init__)
    if "populations" in sig.parameters:
        kwargs["populations"] = 8
    if "population_size" in sig.parameters:
        kwargs["population_size"] = 30

    try:
        model = PySRRegressor(**kwargs)
        model.fit(X, y)
        best = model.get_best()
        y_pred = model.predict(X)
        p = _p90(y_pred, y)
        if verbose:
            print(f"  PySR best: {best['equation']}  p90={p:.2e}")
        if p < 0.05:
            return {"strategy": "pysr", "equation": str(best["equation"]),
                    "complexity": int(best["complexity"]),
                    "loss": float(best["loss"]), "p90": p, "ok": True}
        return {"strategy": "pysr", "equation": str(best["equation"]),
                "p90": p, "ok": False,
                "reason": "p90 above threshold"}
    except Exception as e:
        if verbose:
            traceback.print_exc()
        return {"strategy": "pysr", "error": f"{type(e).__name__}: {str(e)[:200]}",
                "ok": False}


def _try_implicit(formula_str, target_var, n_points=60, p90_thresh=0.05):
    try:
        expr = sp.sympify(formula_str.replace("^", "**"))
    except Exception:
        return None
    target = sp.Symbol(target_var)
    if target not in expr.free_symbols:
        return None
    try:
        sols = sp.solve(sp.Eq(expr, 0), target)
    except Exception:
        return None
    if not sols or len(sols) != 1:
        return None
    sol = sols[0]
    other_syms = [s for s in sol.free_symbols if s != target]
    if not other_syms:
        return None
    vary = other_syms[0]
    fixed_syms = other_syms[1:]
    x_vals = np.logspace(-1, 1, n_points)
    try:
        f = sp.lambdify((vary,) + tuple(fixed_syms), sol, "numpy")
    except Exception:
        return None
    y_vals = []
    for xv in x_vals:
        try:
            yv = float(f(xv, *[1.0] * len(fixed_syms)))
            if not np.isfinite(yv):
                return None
            y_vals.append(yv)
        except Exception:
            return None
    if len(y_vals) < n_points // 2:
        return None
    r = _try_1d(x_vals, np.array(y_vals))
    if r:
        r["strategy"] = "implicit"
        r["target"] = target_var
        r["solved_formula"] = str(sol)
    return r


def discover(x, y, timeout_sec=180, verbose=False):
    x = np.asarray(x, float)
    y = np.asarray(y, float)

    # 1D algebraic: cheapest
    if x.ndim == 1:
        r = _try_1d(x, y)
        if r:
            return r

    # Multivariate: PySR
    if x.ndim == 2:
        r = _try_multivariate_pysr(x, y, timeout_sec=timeout_sec, verbose=verbose)
        if r.get("ok"):
            return r
        return r  # return failure reason so caller sees it

    # 1D fallback: PySR for non-algebraic
    if x.ndim == 1 and HAVE_PYSR:
        r = _try_multivariate_pysr(x.reshape(-1, 1), y,
                                    timeout_sec=timeout_sec, verbose=verbose)
        if r.get("ok"):
            return r

    return {"strategy": "none", "ok": False, "p90": float("inf")}


def capability_report():
    return {
        "1d_algebraic": True,
        "implicit_f0": True,
        "multivariate": HAVE_PYSR,
        "pysr_available": HAVE_PYSR,
        "notes": "Multivariate routes to PySR" if HAVE_PYSR else "pip install pysr for multivariate",
    }
