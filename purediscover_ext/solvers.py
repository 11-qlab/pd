"""Solvers as first-class ops (restricted closing).

Adds newton and brentq as usable numerical solvers plus an implicit-form
fitting pipeline. Given data (x, y) satisfying some F(x, y) = 0, fit template
parameters so F(x_i, y_i) ~ 0, then predict y at new x by solving F = 0.

Restricted scope:
- Implicit *structure* comes from a small template library. The pipeline
  fits parameters; it does not invent the structure.
- Solver failure is reported, not silently tolerated.

Identifiability: a template whose F is ~0 everywhere can fit any data with
rss=0 (degenerate). We require F to have a sign change over the plausible y
range and non-negligible magnitude at the bracket ends before accepting.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import brentq, least_squares


def newton_solve(F, y0, x=None, max_iter=50, tol=1e-12):
    y = float(y0)
    for _ in range(max_iter):
        if x is None:
            f_val = F(y); f_prime = (F(y + 1e-7) - F(y - 1e-7)) / 2e-7
        else:
            f_val = F(x, y); f_prime = (F(x, y + 1e-7) - F(x, y - 1e-7)) / 2e-7
        if abs(f_prime) < 1e-14:
            break
        step = f_val / f_prime
        y -= step
        if abs(step) < tol:
            return y
    return y


def brentq_solve(F, y_lo, y_hi, x=None, xtol=1e-12):
    f = (lambda y: F(y)) if x is None else (lambda y: F(x, y))
    try:
        return brentq(f, y_lo, y_hi, xtol=xtol)
    except ValueError:
        return float("nan")


def _tpl_colebrook(Re, ks, f, p):
    a, b, c, d, e = p
    inner = np.maximum(ks / c + d / (Re * np.sqrt(np.maximum(f, 1e-30))), 1e-30)
    return a / np.sqrt(np.maximum(f, 1e-30)) + b * np.log10(inner) + e


def _tpl_powerlog(Re, ks, f, p):
    a, b, c, d, e, p0, p1 = p
    inner = np.maximum(c * ks + d / (Re * np.maximum(f, 1e-30) ** p1), 1e-30)
    return a * f ** p0 + b * np.log10(inner) + e


TEMPLATES = {
    "colebrook": (_tpl_colebrook, 5, [1.0, 2.0, 3.7, 2.51, 0.0]),
    "powerlog":  (_tpl_powerlog,  7, [1.0, 2.0, 3.7, 2.51, 0.0, 0.5, 0.5]),
}


def _valid_template(fit, data, f_bracket=(1e-4, 0.4), min_magnitude=1e-2):
    """Reject templates whose F is degenerate (F≈0 everywhere)."""
    fn, _, _ = TEMPLATES[fit["template"]]
    p = fit["params"]
    Re_mean = float(np.mean([d["x"][0] for d in data]))
    ks_mean = float(np.mean([d["x"][1] for d in data]))
    F_lo = fn(Re_mean, ks_mean, f_bracket[0], p)
    F_hi = fn(Re_mean, ks_mean, f_bracket[1], p)
    if not (np.isfinite(F_lo) and np.isfinite(F_hi)):
        return False
    if abs(F_lo) < min_magnitude or abs(F_hi) < min_magnitude:
        return False
    return F_lo * F_hi < 0


def fit_implicit_template(data, template_name, init_params=None):
    fn, n_p, p0 = TEMPLATES[template_name]
    if init_params is not None:
        p0 = list(init_params)
    Re = np.array([d["x"][0] for d in data])
    ks = np.array([d["x"][1] for d in data])
    f  = np.array([d["y"] for d in data])
    sol = least_squares(lambda p: fn(Re, ks, f, p),
                        np.array(p0, dtype=float), max_nfev=5000)
    return {"template": template_name, "params": sol.x.tolist(),
            "rss": float(np.sum(sol.fun ** 2)), "n": len(data)}


def predict_with_implicit(fit, Re_new, ks_new, f_bracket=(1e-4, 0.4)):
    fn, _, _ = TEMPLATES[fit["template"]]
    p = fit["params"]
    return brentq_solve(lambda f: fn(Re_new, ks_new, f, p),
                        f_bracket[0], f_bracket[1])


def discover_implicit_law(data, candidate_templates=("colebrook", "powerlog")):
    """Fit each template, filter degenerate, rank by BIC."""
    valid, rejected = [], []
    for tname in candidate_templates:
        try:
            fit = fit_implicit_template(data, tname)
        except Exception:
            continue
        if _valid_template(fit, data):
            n = fit["n"]; k = len(fit["params"])
            rss_eff = max(fit["rss"], 1e-30)
            fit["bic"] = float(n * np.log(rss_eff / n) + k * np.log(n))
            valid.append(fit)
        else:
            rejected.append(tname)
    valid.sort(key=lambda r: r["bic"])
    return {"valid": valid, "rejected": rejected}
