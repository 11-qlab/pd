"""Layer 3 — Bayesian Optimal Experimental Design.

Given competing candidate laws, propose the next x measurement that
maximally discriminates between them.

Score heuristic:
    score(x) = between_var(x) * coverage(x)

    between_var(x) = variance of candidate predictions at x
    coverage(x)    = min distance to existing data points

High when candidates disagree and we haven't measured there.
Ignores uncertainty on individual candidates — refinement is possible
but adds complexity that isn't needed for the demo cases.
"""
from __future__ import annotations

import random

import numpy as np
import sympy as sp


def propose_experiment(candidates, data, var_names, x_range,
                       n_grid: int = 500, top_k: int = 5,
                       min_gap_frac: float = 0.02,
                       random_state: int | None = None) -> float:
    """Return the x value that best discriminates between candidates.

    Args:
        candidates : list of {"expr": sympy_expr, "fit_error": float}
                     (top candidates from discover_and_classify)
        data       : list of (env_dict, y) tuples
        var_names  : variable names (currently supports length 1)
        x_range    : (min, max) for the variable
        n_grid     : resolution of the proposal grid
        top_k      : how many top candidates to consider
        min_gap_frac : min distance (fraction of x_range) from existing x
        random_state : seed for random tie-breaking

    Returns:
        x_next : proposed measurement point
    """
    if len(var_names) != 1:
        raise NotImplementedError("BOED currently supports single-variable data")

    var_name = var_names[0]
    var = sp.Symbol(var_name)
    x_min, x_max = x_range
    rng = random.Random(random_state)

    # Not enough candidates → fall back to random unexplored point
    valid_cands = [
        c for c in candidates
        if c.get("expr") is not None and "fit_error" in c
    ]
    if len(valid_cands) < 2:
        return _random_unexplored(data, var_name, x_range, rng)

    valid_cands = sorted(valid_cands, key=lambda c: c["fit_error"])[:top_k]

    # Grid of proposal points
    x_grid = np.linspace(x_min, x_max, n_grid)

    # Predictions per candidate
    predictions = []
    for c in valid_cands:
        y_pred = []
        ok = True
        for xv in x_grid:
            try:
                yv = float(c["expr"].subs({var: float(xv)}))
                if not np.isfinite(yv):
                    ok = False
                    break
                y_pred.append(yv)
            except Exception:
                ok = False
                break
        if ok and len(y_pred) == n_grid:
            predictions.append(np.array(y_pred))

    if len(predictions) < 2:
        return _random_unexplored(data, var_name, x_range, rng)

    preds = np.array(predictions)  # (n_cands, n_grid)

    # Between-candidate variance
    between = np.var(preds, axis=0)
    if between.max() <= 0:
        return _random_unexplored(data, var_name, x_range, rng)

    # Coverage: distance from x_grid to nearest existing data point
    existing_x = np.array([env[var_name] for env, _ in data], dtype=float)
    coverage = np.array([
        np.min(np.abs(x - existing_x)) for x in x_grid
    ])

    # Normalize
    between_n = between / (between.max() + 1e-12)
    if coverage.max() > 0:
        coverage_n = coverage / coverage.max()
    else:
        coverage_n = np.zeros_like(coverage)

    score = between_n * coverage_n

    # Penalize x too close to existing points
    min_gap = min_gap_frac * (x_max - x_min)
    score[coverage < min_gap] = 0.0

    if np.max(score) <= 0:
        return _random_unexplored(data, var_name, x_range, rng)

    # Slight random tie-break to avoid always picking the same x
    top_indices = np.argsort(score)[-5:]
    chosen = rng.choice(top_indices)
    return float(x_grid[chosen])


def _random_unexplored(data, var_name, x_range, rng, n_tries: int = 50):
    """Pick a random x that's not too close to any existing point."""
    x_min, x_max = x_range
    existing = [env[var_name] for env, _ in data]
    min_gap = 0.05 * (x_max - x_min)
    for _ in range(n_tries):
        x = rng.uniform(x_min, x_max)
        if all(abs(x - e) >= min_gap for e in existing):
            return x
    return rng.uniform(x_min, x_max)


__all__ = ["propose_experiment"]
