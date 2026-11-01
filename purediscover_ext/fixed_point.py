"""Fixed-point iteration discovery for implicit laws (Colebrook testbed).

Colebrook: 1/sqrt(f) = -2*log10(ks_D/3.7 + 2.51/(Re*sqrt(f)))
Rearranged: f_new = g(Re, ks_D, f_old), iterated to a fixed point.

Note on identifiability: the log-linear form
    a*log10(b*ks + c/(Re*sqrt(f))) + d
has a scale degeneracy. Only the ratio c/b is identifiable when d is free.
We constrain d = 0 to break the degeneracy and recover the canonical
coefficients (a=-2, b=1/3.7, c=2.51).
"""
from __future__ import annotations
import numpy as np


def colebrook_update(Re, ks_D, f):
    inner = ks_D / 3.7 + 2.51 / (Re * np.sqrt(np.maximum(f, 1e-12)))
    denom = -2.0 * np.log10(np.maximum(inner, 1e-30))
    return 1.0 / (denom ** 2)


def colebrook_update_vec(X, y):
    Re, ks_D = X[..., 0], X[..., 1]
    inner = ks_D / 3.7 + 2.51 / (Re * np.sqrt(np.maximum(y, 1e-12)))
    denom = -2.0 * np.log10(np.maximum(inner, 1e-30))
    return 1.0 / (denom ** 2)


def colebrook_solve(Re, ks_D, f0=0.02, tol=1e-10, max_iter=50):
    f = f0
    for _ in range(max_iter):
        f_new = colebrook_update(Re, ks_D, f)
        if abs(f_new - f) < tol:
            return f_new
        f = f_new
    return f


def generate_iteration_traces(n_pts=1000, seed=0):
    rng = np.random.default_rng(seed)
    Re = 10 ** rng.uniform(4, 7, n_pts)
    ks_D = 10 ** rng.uniform(-5, -1.5, n_pts)
    f_old = rng.uniform(0.01, 0.06, n_pts)
    f_new = colebrook_update_vec(np.stack([Re, ks_D], axis=-1), f_old)
    return {"Re": Re, "ks_D": ks_D, "f_old": f_old, "f_new": f_new}


def _try_purediscover_fit(X, y, var_names):
    for mod_path, fn_name in [
        ("purediscover", "discover"),
        ("purediscover", "discover_expression"),
        ("purediscover.discovery", "discover"),
        ("purediscover.core", "discover"),
    ]:
        try:
            mod = __import__(mod_path, fromlist=[fn_name])
            fn = getattr(mod, fn_name, None)
            if fn is None:
                continue
            try:
                return fn(X, y, var_names=var_names)
            except TypeError:
                return fn(X, y)
        except (ImportError, Exception):
            continue
    return None


def discover_update_rule(data):
    """Discover g(Re, ks_D, f_old) -> f_new."""
    X = np.column_stack([data["Re"], data["ks_D"], data["f_old"]])
    y = data["f_new"]

    expr = _try_purediscover_fit(X, y, ["Re", "ks_D", "f_old"])
    if expr is not None:
        return {"kind": "symbolic", "expr": expr}

    # Fallback with d=0 to break the scale degeneracy.
    from scipy.optimize import least_squares
    Re, ks_D, f_old, f_new = data["Re"], data["ks_D"], data["f_old"], data["f_new"]
    inv_sqrt_f = 1.0 / np.sqrt(f_new)
    sqrt_f_old = np.sqrt(f_old)

    def residual(p):
        a, b, c = p
        inner = b * ks_D + c / (Re * sqrt_f_old)
        return a * np.log10(np.maximum(inner, 1e-30)) - inv_sqrt_f

    x0 = np.array([-1.5, 0.2, 1.5])
    sol = least_squares(residual, x0, max_nfev=4000)
    a, b, c = sol.x

    def update(Re_, ks_D_, f_old_):
        inner = b * ks_D_ + c / (Re_ * np.sqrt(np.maximum(f_old_, 1e-12)))
        return 1.0 / (a * np.log10(np.maximum(inner, 1e-30))) ** 2

    return {
        "kind": "parametric",
        "params": {"a": a, "b": b, "c": c, "d": 0.0},
        "update": update,
        "residual_rms": float(np.sqrt(np.mean(sol.fun ** 2))),
    }


def solve_with_discovered(rule, Re, ks_D, f0=0.02, tol=1e-10, max_iter=50):
    f = f0
    for _ in range(max_iter):
        f_new = rule["update"](Re, ks_D, f)
        if abs(f_new - f) < tol:
            return f_new
        f = f_new
    return f
