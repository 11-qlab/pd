"""Layer 1 — statistical robustness.

Three sub-checks, all fast (~200ms per candidate):
    cross-validation : evaluate on held-out 20%, compare to train error
    bootstrap        : evaluate on 50 resamples, measure stability
    parsimony (BIC)  : penalize free parameters

Each returns a sub-dict; the layer as a whole returns .ok = all(sub.ok).
"""
from __future__ import annotations
import math
import random

import sympy as sp


def _fit_error(expr, data) -> float:
    mags = sorted(abs(y) for _, y in data if abs(y) > 0)
    y_scale = mags[len(mags) // 2] if mags else 1.0
    floor = max(y_scale / 20.0, 1e-300)
    errs = []
    for env, y in data:
        try:
            yp = float(expr.subs({sp.Symbol(k): v for k, v in env.items()}))
        except (ZeroDivisionError, ValueError, TypeError):
            return 1.0
        if not math.isfinite(yp):
            return 1.0
        errs.append(abs(y - yp) / max(abs(y), floor))
    return sum(errs) / max(1, len(errs))


def _cv_gate(expr, data, *, k=5, seed=0, max_ratio=3.0):
    rng = random.Random(seed)
    idx = list(range(len(data)))
    rng.shuffle(idx)
    fold_size = max(1, len(data) // k)
    ratios = []
    for f in range(k):
        start = f * fold_size
        end = start + fold_size if f < k - 1 else len(data)
        test_i = idx[start:end]
        train_i = idx[:start] + idx[end:]
        if not test_i or not train_i:
            continue
        train = [data[i] for i in train_i]
        test = [data[i] for i in test_i]
        train_err = _fit_error(expr, train)
        test_err = _fit_error(expr, test)
        # Accept when EITHER absolute error is small OR test/train is
        # within max_ratio. Prevents rejection of near-perfect fits
        # (train_err ~ 1e-5, test_err ~ 2e-4 → ratio 20 but both fine).
        if test_err <= max(0.02, max_ratio * train_err):
            ratios.append(1.0)   # treat as passing fold
        else:
            ratios.append(test_err / max(train_err, 1e-6))
    if not ratios:
        return {"k": k, "ratios": [], "max_ratio": 0.0,
                "ok": False, "reason": "no folds"}
    max_r = max(ratios)
    return {
        "k": k,
        "ratios": ratios,
        "max_ratio": max_r,
        "threshold": max_ratio,
        "ok": max_r <= max_ratio,
    }


def _bootstrap_gate(expr, data, *, n=50, seed=0, factor=2.5):
    rng = random.Random(seed)
    n_data = len(data)
    base_err = _fit_error(expr, data)
    threshold = max(0.05, factor * base_err)
    errs = []
    stable = 0
    for _ in range(n):
        idx = [rng.randrange(n_data) for _ in range(n_data)]
        sample = [data[i] for i in idx]
        e = _fit_error(expr, sample)
        errs.append(e)
        if e <= threshold:
            stable += 1
    frac = stable / max(1, n)
    return {
        "n": n,
        "base_err": base_err,
        "threshold": threshold,
        "stable_count": stable,
        "stable_fraction": frac,
        "errs": errs,
        "ok": frac >= 0.6,
    }


def _bic(expr, data, var_names) -> dict:
    n = len(data)
    errs_sq = []
    for env, y in data:
        try:
            yp = float(expr.subs({sp.Symbol(k): v for k, v in env.items()}))
        except Exception:
            return {"value": float("inf"), "n_params": 0,
                    "n_data": n, "ok": False}
        errs_sq.append((yp - y) ** 2)
    rss = sum(errs_sq)
    if rss <= 0:
        rss = 1e-30
    k_params = len([s for s in expr.free_symbols if str(s) not in var_names])
    k_params = max(1, k_params)
    value = n * math.log(rss / n) + k_params * math.log(n)
    return {"value": value, "n_params": k_params, "n_data": n, "ok": True}


def statistical_gate(expr, data, var_names, *,
                     n_bootstrap=50, k_folds=5, seed=0) -> dict:
    """Layer 1: CV + bootstrap + BIC, combined."""
    cv = _cv_gate(expr, data, k=k_folds, seed=seed)
    bs = _bootstrap_gate(expr, data, n=n_bootstrap, seed=seed)
    bic = _bic(expr, data, var_names)
    ok = cv["ok"] and bs["ok"] and bic["ok"]
    return {
        "ok": ok,
        "cv": cv,
        "bootstrap": bs,
        "bic": bic,
    }
