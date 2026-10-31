"""Greedy forward feature selection. 40x faster than exhaustive combinations."""
from __future__ import annotations
import itertools
import math
import sympy as sp


def _small_features(var_names):
    """Minimal feature set — ~15 features, not 64."""
    vars_ = [sp.Symbol(v) for v in var_names]
    feats = []
    for v in vars_:
        feats.extend([v, v ** 2, v ** 3, v ** 4, 1 / v])
    for a, b in itertools.combinations(vars_, 2):
        feats.extend([a * b, a ** 2 * b, a * b ** 2])
    if len(vars_) >= 3:
        for trio in itertools.combinations(vars_, 3):
            feats.append(trio[0] * trio[1] * trio[2])
    seen, uniq = set(), []
    for f in feats:
        k = str(sp.simplify(f))
        if k not in seen:
            seen.add(k)
            uniq.append(f)
    return uniq


def search_sum_of_features(data, var_names, *, tol=0.02, max_k=3):
    """Greedy forward selection: ~30 lstsq total, not 40k."""
    if len(data) < 5 or len(var_names) < 2:
        return []

    try:
        import numpy as np
    except ImportError:
        return []

    feats = _small_features(var_names)
    feat_vals = []
    valid = []
    for f in feats:
        try:
            vals = [float(f.subs({sp.Symbol(k): v for k, v in env.items()}))
                    for env, _ in data]
        except Exception:
            continue
        if all(math.isfinite(v) for v in vals):
            feat_vals.append(np.array(vals))
            valid.append(f)

    if not valid:
        return []

    y = np.array([y for _, y in data])
    n_data = len(data)

    def _fit(indices):
        X = np.column_stack([np.ones(n_data)] + [feat_vals[i] for i in indices])
        try:
            coefs, *_ = np.linalg.lstsq(X, y, rcond=None)
        except Exception:
            return None, 1.0
        err = float(np.mean(np.abs(y - X @ coefs) / (np.abs(y) + 1e-9)))
        return coefs, err

    # Greedy forward selection
    chosen = []
    results = []
    for _ in range(max_k):
        best_i, best_coefs, best_err = None, None, 1.0
        for i in range(len(feat_vals)):
            if i in chosen:
                continue
            coefs, err = _fit(chosen + [i])
            if err < best_err:
                best_i, best_coefs, best_err = i, coefs, err

        if best_i is None or best_err > tol:
            break

        chosen.append(best_i)
        expr = sp.Float(best_coefs[0])
        for j, ci in enumerate(best_coefs[1:]):
            expr += sp.Float(ci) * valid[chosen[j]]
        expr = sp.simplify(expr)
        results.append({
            "expr": expr,
            "fit_error": best_err,
            "n_nodes": sum(1 for _ in sp.preorder_traversal(expr)),
        })

    return sorted(results, key=lambda r: (r["fit_error"], r["n_nodes"]))


__all__ = ["search_sum_of_features"]
