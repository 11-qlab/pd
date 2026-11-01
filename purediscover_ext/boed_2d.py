"""Disagreement-based active proposal for 2D discovery loops.

Among the top-K surviving candidates, find the candidate grid point where
they disagree most. That point is the highest-information next measurement.

Cheap approximation of BOED: no posterior needed, just ensemble disagreement.
"""
from __future__ import annotations

import math

import numpy as np
import sympy as sp


def propose_boed_2d(candidates, data, var_names, x_range, y_range,
                    n_grid=40, top_k=5, min_gap=0.02):
    if len(var_names) != 2:
        raise ValueError(f"needs 2 vars, got {var_names}")
    if not candidates:
        rng = np.random.default_rng(0)
        return {"x": float(rng.uniform(*x_range)),
                "y": float(rng.uniform(*y_range))}

    # take top-K by MDL bits (already sorted by caller) with finite bits
    valid = [c for c in candidates
             if math.isfinite(float(c.get("bits", c.get("mdl_bits", float("inf")))))]
    valid = valid[:top_k]
    if not valid:
        rng = np.random.default_rng(0)
        return {"x": float(rng.uniform(*x_range)),
                "y": float(rng.uniform(*y_range))}

    # if only one candidate survives, fall back to uniform sampling
    if len(valid) == 1:
        rng = np.random.default_rng(0)
        pts = [(d[0][var_names[0]], d[0][var_names[1]]) for d in data]
        for _ in range(500):
            p = (float(rng.uniform(*x_range)), float(rng.uniform(*y_range)))
            if all(abs(p[0]-q[0])>min_gap or abs(p[1]-q[1])>min_gap for q in pts):
                return {"x": p[0], "y": p[1]}
        return {"x": float(rng.uniform(*x_range)), "y": float(rng.uniform(*y_range))}

    x_sym = sp.Symbol(var_names[0])
    y_sym = sp.Symbol(var_names[1])
    # extract sympy expressions from candidates
    exprs = []
    for c in valid:
        e = c.get("sympy_expr") or c.get("expr")
        if isinstance(e, str):
            try:
                e = sp.sympify(e)
            except Exception:
                continue
        if isinstance(e, sp.Expr):
            exprs.append(e)
    if len(exprs) < 2:
        rng = np.random.default_rng(0)
        return {"x": float(rng.uniform(*x_range)),
                "y": float(rng.uniform(*y_range))}

    grid_x = np.linspace(*x_range, n_grid)
    grid_y = np.linspace(*y_range, n_grid)
    existing = [(d[0][var_names[0]], d[0][var_names[1]]) for d in data]

    best_point, best_score = None, -1.0
    for xi in grid_x:
        for yi in grid_y:
            if any(abs(xi-q[0])<min_gap and abs(yi-q[1])<min_gap for q in existing):
                continue
            preds = []
            for e in exprs:
                try:
                    f = sp.lambdify([x_sym, y_sym], e, modules=["numpy"])
                    v = float(f(xi, yi))
                    if math.isfinite(v):
                        preds.append(v)
                except Exception:
                    continue
            if len(preds) < 2:
                continue
            # disagreement score: max - min among predictions
            score = max(preds) - min(preds)
            if score > best_score:
                best_score, best_point = score, (float(xi), float(yi))

    if best_point is None:
        rng = np.random.default_rng(0)
        return {"x": float(rng.uniform(*x_range)),
                "y": float(rng.uniform(*y_range))}
    return {"x": best_point[0], "y": best_point[1], "disagreement": best_score}
