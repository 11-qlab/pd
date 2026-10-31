"""Numeric convergence check with DATA-DRIVEN corners.

Corners for each variable are {min, median, max} from the actual data,
not a fixed abstract range. This is critical: laws like arrhenius
(T ∈ [280, 400]) or yukawa (lam ∈ [0.5, 5]) have narrow ranges where
extrapolating to {0.2, 20} makes exponentials diverge — rejecting
valid fits. Data-driven corners test the parameter space the data
actually covers.

A surrogate that fits training data but diverges outside it will
still fail because {min, mid, max} spans the full observed range.
"""
from __future__ import annotations

import math
from itertools import product

import sympy as sp


class ConvergenceChecker:
    def __init__(self, atol: float = 0.05, min_samples: int = 3,
                 timeout_s: float | None = None):
        self.atol = atol
        self.min_samples = min_samples

    def _data_corners(self, data, syms) -> list[float]:
        """For each symbol, return {min, mid, max} from data."""
        corners_per_var = []
        for s in syms:
            values = [env.get(str(s)) for env, _ in data]
            values = [v for v in values if v is not None and math.isfinite(v)]
            if not values:
                # Fallback if symbol not in data (shouldn't happen)
                corners_per_var.append((1.0,))
                continue
            lo = min(values)
            hi = max(values)
            mid = (lo + hi) / 2
            # Dedupe near-identical values
            distinct = sorted(set([lo, mid, hi]))
            if len(distinct) == 1:
                corners_per_var.append((lo,))
            elif len(distinct) == 2:
                corners_per_var.append((distinct[0], distinct[1]))
            else:
                corners_per_var.append(tuple(distinct))
        return corners_per_var

    def is_converged(self, found, truth, data=None) -> bool:
        if found is None or truth is None:
            return False
        if found == truth:
            return True
        if not data:
            return False

        try:
            ratio_expr = found / truth
        except Exception:
            return False

        ratios: list[float] = []

        # Data points
        for env, _ in data:
            try:
                sub = {sp.Symbol(k): float(v) for k, v in env.items()}
                r = float(ratio_expr.subs(sub))
            except Exception:
                continue
            if math.isfinite(r):
                ratios.append(r)

        # Data-driven corners: {min, mid, max} for each variable
        syms = sorted(found.free_symbols | truth.free_symbols, key=str)
        if syms and data:
            corners_per_var = self._data_corners(data, syms)
            for combo in product(*corners_per_var):
                sub = {s: v for s, v in zip(syms, combo)}
                try:
                    r = float(ratio_expr.subs(sub))
                except Exception:
                    continue
                if math.isfinite(r):
                    ratios.append(r)

        if len(ratios) < self.min_samples:
            return False

        r0 = ratios[0]
        if abs(r0) < 1e-6:
            return False

        return all(
            abs(r - r0) <= self.atol * max(1.0, abs(r0))
            for r in ratios
        )


__all__ = ["ConvergenceChecker"]
