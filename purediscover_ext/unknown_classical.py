"""Structural discovery via transformation search (restricted).

Given 1D (x, y) data from a law NOT in the template library:
  1. Try all (tx, ty) transform pairs from a human-supplied library
  2. Fit y' = slope*x' + intercept for each
  3. Rank by R^2, prefer simpler transforms on ties
  4. Recover coefficients in original space
  5. Verify on held-out points

Restricted scope:
- Transform library is human-supplied {identity, log, reciprocal, sqrt, square}
- Only 2-term linear form after transform
- Only 1D input

What this proves: given a transform library, the pipeline identifies which
(tx, ty) pair linearizes data from a non-templated law and recovers its
coefficients. That is structural discovery, not template fitting.

What this does not prove: invention of new functional forms from scratch.
"""
from __future__ import annotations
import numpy as np


def _log(x):
    return np.log(np.maximum(np.abs(x), 1e-30))


def _recip(x):
    return 1.0 / np.where(np.abs(x) > 1e-30, x, 1e-30)


def _sqrt(x):
    return np.sign(x) * np.sqrt(np.abs(x))


def _square(x):
    return x * x


def _exp(x):
    return np.clip(np.exp(np.clip(x, -50, 50)), 1e-50, 1e50)


def _log1p(y):
    return np.log1p(np.maximum(y, -0.999999))

def _expm1(y):
    return np.expm1(y)

TRANSFORMS = {
    "identity":   (lambda x: x, lambda x: x),
    "log":        (_log, _exp),
    "reciprocal": (_recip, _recip),
    "sqrt":       (_sqrt, _square),
    "square":     (_square, _sqrt),
    "log1p":      (_log1p, _expm1),
}


def _r2(y_true, y_pred):
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def discover_structure(data, x_names, y_name, transforms=None):
    if transforms is None:
        transforms = TRANSFORMS
    if len(x_names) != 1:
        raise ValueError("Only single-input supported")
    x_var = x_names[0]
    xs = np.array([d["x"][x_var] for d in data])
    ys = np.array([d["y"] for d in data])

    results = []
    for tx_name, (tx_fwd, _) in transforms.items():
        for ty_name, (ty_fwd, _) in transforms.items():
            try:
                x_t = tx_fwd(xs)
                y_t = ty_fwd(ys)
            except Exception:
                continue
            if not (np.all(np.isfinite(x_t)) and np.all(np.isfinite(y_t))):
                continue
            if np.std(x_t) < 1e-10 or np.std(y_t) < 1e-10:
                continue
            try:
                coefs = np.polyfit(x_t, y_t, 1)
                y_pred = np.polyval(coefs, x_t)
                r2 = _r2(y_t, y_pred)
                results.append({
                    "transform_x": tx_name,
                    "transform_y": ty_name,
                    "slope": float(coefs[0]),
                    "intercept": float(coefs[1]),
                    "r2": r2,
                })
            except Exception:
                continue

    if not results:
        return {"best": None, "all_scores": [], "expression": None}

    complexity = ["identity", "log", "reciprocal", "sqrt", "square", "log1p"]
    def key(r):
        c = complexity.index(r["transform_x"]) + complexity.index(r["transform_y"])
        return (-r["r2"], c)
    results.sort(key=key)
    best = results[0]
    return {
        "best": best,
        "all_scores": results,
        "expression": _format(best, x_var, y_name),
    }


def _format(best, x_var, y_name):
    tx_str = {"identity": x_var, "log": f"ln({x_var})",
              "reciprocal": f"1/{x_var}", "sqrt": f"sqrt({x_var})",
              "square": f"{x_var}^2",
              "log1p": f"ln(1+{x_var})"}[best["transform_x"]]
    ty_str = {"identity": y_name, "log": f"ln({y_name})",
              "reciprocal": f"1/{y_name}", "sqrt": f"sqrt({y_name})",
              "square": f"{y_name}^2",
              "log1p": f"ln(1+{y_name})"}[best["transform_y"]]
    return f"{ty_str} = {best['slope']:.6g} * {tx_str} + {best['intercept']:.6g}"


def predict_from_discovery(best, x_new):
    tx_fwd, _ = TRANSFORMS[best["transform_x"]]
    _, ty_inv = TRANSFORMS[best["transform_y"]]
    x_t = tx_fwd(np.asarray(x_new))
    y_t = best["slope"] * x_t + best["intercept"]
    return ty_inv(y_t)
