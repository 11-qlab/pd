"""Exponential-form discovery via constrained log-linear regression.

v3 changelog:
    1. GREEDY OVER COMPLEXITY. Try k=1 first. If the best k=1 fit is
       already < 0.5*tol, return it. Only try k=2 if k=1 failed.
       This makes clean Arrhenius (1/T) win over degenerate 2-feature
       approximations of it (T^48 * exp(-0.07*T)).

    2. CAP ON LOG-FEATURE COEFFICIENT. |c_log| <= 20. Real power
       laws have exponents 1-6 (Stefan-Boltzmann T^4, Debye T^3).
       T^48 is a signature of degeneracy, not physics.
"""
from __future__ import annotations
import itertools
import math

import sympy as sp


MAX_COEFF = 1e4
MAX_ARG_RANGE = 50.0
MAX_LOG_COEFF = 20.0
MAX_FEATURES = 2


def _fit_linear(features, y):
    try:
        import numpy as np
    except ImportError:
        return None
    X = np.column_stack([np.ones(len(y))] + features)
    coefs, *_ = np.linalg.lstsq(X, np.array(y), rcond=None)
    resid = float(np.linalg.norm(X @ coefs - np.array(y)))
    return coefs, resid


def _fit_error(expr, data) -> float:
    errs = []
    for env, y in data:
        try:
            yp = float(expr.subs({sp.Symbol(k): v for k, v in env.items()}))
        except (ZeroDivisionError, ValueError, TypeError):
            return 1.0
        if not math.isfinite(yp):
            return 1.0
        errs.append(abs(y - yp) / (abs(y) + 1e-9))
    return sum(errs) / max(1, len(errs))


def _feature_functions(var_names):
    out = []
    for v in var_names:
        out.append((f"{v}",      lambda env, v=v: env[v]))
        out.append((f"1/{v}",    lambda env, v=v: 1.0 / env[v] if env[v] != 0 else float("nan")))
        out.append((f"{v}^2",    lambda env, v=v: env[v] ** 2))
        out.append((f"log({v})", lambda env, v=v: math.log(env[v]) if env[v] > 0 else float("nan")))
        out.append((f"sqrt({v})",lambda env, v=v: math.sqrt(env[v]) if env[v] >= 0 else float("nan")))
    return out


def _feature_to_sympy(name: str):
    if "/" in name:
        return sp.Symbol(name.split("/")[1].strip()) ** (-1)
    if "^2" in name:
        return sp.Symbol(name.replace("^2", "").strip()) ** 2
    if name.startswith("log("):
        return sp.log(sp.Symbol(name[4:-1].strip()))
    if name.startswith("sqrt("):
        return sp.sqrt(sp.Symbol(name[5:-1].strip()))
    return sp.Symbol(name)


def _snap(c, eps=0.02):
    from fractions import Fraction
    if abs(c) < 1e-9:
        return sp.Integer(0)
    frac = Fraction(float(c)).limit_denominator(1000)
    if abs(frac.numerator) > 10000:
        return sp.Float(c)
    if abs(c - float(frac)) < eps * max(1.0, abs(c)):
        return sp.Rational(frac.numerator, frac.denominator)
    return sp.Float(c)


def _arg_range(coefs, features_vals, n_data):
    args = []
    for i in range(n_data):
        a = coefs[0]
        for j, vals in enumerate(features_vals):
            a += coefs[j + 1] * vals[i]
        args.append(a)
    return max(args) - min(args)


def _bic(resid, n, k):
    if resid <= 0 or n <= 0:
        return float("inf")
    rss = resid * resid
    return n * math.log(max(rss, 1e-300) / n) + k * math.log(n)


def _constraints_ok(coefs, chosen_names, chosen_vals, n):
    """Reject fits that are numerically possible but physically absurd."""
    if any(abs(c) > MAX_COEFF for c in coefs[1:]):
        return False
    if _arg_range(coefs, chosen_vals, n) > MAX_ARG_RANGE:
        return False
    for c, name in zip(coefs[1:], chosen_names):
        if name.startswith("log(") and abs(c) > MAX_LOG_COEFF:
            return False
    return True


def search_exponential(data, var_names, *, tol: float = 0.02,
                       max_features: int = MAX_FEATURES):
    """Find y = A*exp(f), f linear in small feature set.

    Returns fits from the SMALLEST k where any candidate passes
    fit_error < 0.5*tol. Falls back to all fits if none do.
    """
    if not data or len(data) < 3 or not all(y > 0 for _, y in data):
        return []

    log_data = [(env, math.log(y)) for env, y in data]
    log_y = [ly for _, ly in log_data]
    n = len(log_y)

    feature_vals = []
    valid_names = []
    for name, fn in _feature_functions(var_names):
        try:
            vals = [fn(env) for env, _ in log_data]
        except Exception:
            continue
        if any(math.isnan(v) or math.isinf(v) for v in vals):
            continue
        feature_vals.append(vals)
        valid_names.append(name)
    if not valid_names:
        return []

    fits_by_k: dict[int, list[dict]] = {}

    for k in range(1, min(max_features, len(valid_names)) + 1):
        results_k: list[dict] = []
        for combo in itertools.combinations(range(len(valid_names)), k):
            chosen_vals = [feature_vals[i] for i in combo]
            chosen_names = [valid_names[i] for i in combo]

            fit = _fit_linear(chosen_vals, log_y)
            if fit is None:
                continue
            coefs, resid = fit

            if not _constraints_ok(coefs, chosen_names, chosen_vals, n):
                continue

            # Split the constant term out of the exponent so that
            # exp(c0 + c1/T) becomes A*exp(c1/T) with A = exp(c0) —
            # which is how the law is conventionally written.
            const_term = _snap(coefs[0])
            var_arg = sp.Integer(0)
            ok = True
            for c, name in zip(coefs[1:], chosen_names):
                sym = _feature_to_sympy(name)
                if sym is None:
                    ok = False
                    break
                var_arg += _snap(c) * sym
            if not ok:
                continue
            try:
                A_val = float(sp.exp(const_term))
                if 1e-300 < A_val < 1e300:
                    A_expr = sp.Float(A_val)
                else:
                    A_expr = sp.exp(const_term)
                expr = sp.simplify(A_expr * sp.exp(var_arg))
            except Exception:
                continue

            fe = _fit_error(expr, data)
            if fe > tol:
                continue

            results_k.append({
                "expr": expr,
                "feature_combo": chosen_names,
                "fit_error": fe,
                "bic": _bic(resid, n, k + 1),
                "k": k,
            })

        if results_k:
            fits_by_k[k] = results_k
            # GREEDY: stop at first k that produces a good fit
            if min(r["fit_error"] for r in results_k) < 0.5 * tol:
                break

    # Return the fits from the smallest successful k
    for k in sorted(fits_by_k):
        return sorted(fits_by_k[k], key=lambda r: (r["fit_error"], r["bic"]))
    return []
