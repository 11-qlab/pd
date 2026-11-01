"""Sparse regression arm with relative-MRE filter before MDL scoring."""
from __future__ import annotations

import itertools
import math
import time
import warnings

import numpy as np
import sympy as sp

from purediscover.mdl import mdl_score


def _extend_with_trig(feats, x_syms, include_trig=True):
    """Add composite transcendental features: cos/sin/exp on products and singles.

    This is what makes the sparse arm reachable for cos(2*x*y)-type laws.
    Enumerates k in {1, 2} for frequencies; exp uses -z^2.
    """
    if not include_trig:
        return feats
    n = len(x_syms)
    extra = []
    # single-variable trig
    for v in x_syms:
        for k in (1, 2):
            extra.append(sp.cos(k * v))
            extra.append(sp.sin(k * v))
        extra.append(sp.exp(-(v ** 2)))
    # pairwise product trig (this catches cos(2*x*y))
    for i in range(n):
        for j in range(i + 1, n):
            z = x_syms[i] * x_syms[j]
            for k in (1, 2):
                extra.append(sp.cos(k * z))
                extra.append(sp.sin(k * z))
            extra.append(sp.exp(-(z ** 2)))
    return feats + extra


def _build_features(x_syms, degree=2, include_ratio=True, include_trig=True):
    feats = [sp.Integer(1)]
    feats.extend(x_syms)
    n = len(x_syms)
    for i in range(n):
        for j in range(i, n):
            feats.append(x_syms[i] * x_syms[j])
    if degree >= 3:
        for i in range(n):
            for j in range(i, n):
                for k in range(j, n):
                    feats.append(x_syms[i] * x_syms[j] * x_syms[k])
    if include_ratio:
        for i in range(n):
            for j in range(n):
                if i != j:
                    feats.append(x_syms[i] / (x_syms[j] + 1))
    return _extend_with_trig(feats, x_syms, include_trig=include_trig)


def _evaluate_basis(feats, x_syms, data):
    xs = {v: np.array([d[0][str(v)] for d in data], dtype=float) for v in x_syms}
    ys = np.array([d[1] for d in data], dtype=float)
    cols = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with np.errstate(all="ignore"):
            for f in feats:
                try:
                    fn = sp.lambdify(list(x_syms), f, modules=["numpy"])
                    col = fn(*[xs[v] for v in x_syms])
                    col = np.asarray(col, dtype=float)
                    if col.shape != ys.shape:
                        col = np.broadcast_to(col, ys.shape).copy()
                    if not np.all(np.isfinite(col)):
                        return None
                    cols.append(col)
                except Exception:
                    return None
    return np.column_stack(cols), ys


def tree_search_ext(
    data,
    var_names,
    *,
    max_features: int = 4,
    degree: int = 2,
    include_trig: bool = True,
    tol: float = 0.02,
    time_budget: float = 60.0,
    accept_mre_factor: float = 10.0,
    top_k_per_size: int = 30,
    relative_mre_gate: float = 5.0,
):
    """Sparse regression with a relative-MRE gate before MDL.

    relative_mre_gate: candidates with MRE > gate * (best MRE at this k)
    are dropped before mdl_score sees them.  Without this, mdl_score's
    structural-bits term can win for a 5-node 18%-MRE form over a
    7-node 1%-MRE form, because residual encoding is underweighted.
    """
    t0 = time.time()
    x_syms = [sp.Symbol(v) for v in var_names]
    feats = _build_features(x_syms, degree=degree, include_trig=include_trig)

    built = _evaluate_basis(feats, x_syms, data)
    if built is None:
        return []
    A, ys = built
    n_feat = A.shape[1]
    accept_mre = tol * accept_mre_factor

    candidates = []
    max_features = min(max_features, n_feat - 1)

    for k in range(1, max_features + 1):
        if time.time() - t0 > time_budget * 0.9:
            break

        fits = []
        for combo in itertools.combinations(range(n_feat), k):
            sub = A[:, combo]
            try:
                coef, *_ = np.linalg.lstsq(sub, ys, rcond=None)
            except np.linalg.LinAlgError:
                continue
            yp = sub @ coef
            if not np.all(np.isfinite(yp)):
                continue
            mre = float(np.mean(np.abs(yp - ys) / (np.abs(ys) + 1e-12)))
            if not math.isfinite(mre) or mre > accept_mre:
                continue
            fits.append((mre, combo, coef))

        if not fits:
            continue

        fits.sort(key=lambda t: t[0])
        best_mre_at_k = fits[0][0]

        # Relative gate: reject candidates > gate * best at this k.
        # This is what removes the 18%-MRE quadratic candidate before
        # mdl_score can arbitrate incorrectly in its favor.
        fits = [f for f in fits
                if f[0] <= relative_mre_gate * best_mre_at_k + 1e-9]
        fits = fits[:top_k_per_size]

        for mre, combo, coef in fits:
            if time.time() - t0 > time_budget:
                break
            expr = sp.Integer(0)
            for c, idx in zip(coef, combo):
                if abs(c) < 1e-12:
                    continue
                expr += float(c) * feats[idx]
            try:
                s = mdl_score(expr, data,
                              residual_tol=max(mre * 3.0, 0.05))
                if not isinstance(s, dict) or not s.get("ok"):
                    continue
                bits = float(s.get("bits", float("inf")))
                if not math.isfinite(bits):
                    continue
                candidates.append({
                    "expr": str(expr),
                    "sympy_expr": expr,
                    "bits": bits,
                    "mre": mre,
                    "n_nodes": int(k),
                    "class": "NOVEL",
                    "fit_error": mre,
                    "coeffs": coef.tolist(),
                })
            except Exception:
                pass

        if time.time() - t0 > time_budget:
            break

    candidates.sort(key=lambda c: c["bits"])
    return candidates
