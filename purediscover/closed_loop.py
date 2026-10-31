"""Layer 4 — Closed-loop discovery.

The outer loop that turns single-shot discovery into autonomous
experimental design:

    1. Discover candidates from current data
    2. Check convergence (top candidate matches ground truth form)
    3. Propose next experiment (BOED)
    4. Simulate measurement at proposed x
    5. Append to data, repeat
"""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field

import numpy as np
import sympy as sp

from purediscover.discover import discover_and_classify
from purediscover.experiment_design import propose_experiment


def _equivalent_up_to_scalar(a, b, data=None, atol: float = 0.05) -> bool:
    """True iff a and b are equivalent up to a constant multiplier.

    If data is given, sample at the data points (correct range).
    Otherwise fall back to a wider log-spaced grid to reduce underflow.

    Guards against the degenerate 0/X case: a=0, b=exp(...) passes
    "ratio is constant" through simplify(a/b) but is meaningless.
    """
    if a is None or b is None:
        return False

    try:
        if sp.simplify(a - b) == 0:
            return True
    except Exception:
        pass

    # Build sample points
    samples = []
    if data is not None:
        for env, _ in data:
            samples.append({sp.Symbol(k): v for k, v in env.items()})
    else:
        syms = list(a.free_symbols | b.free_symbols)
        if not syms:
            return False
        import random as _r
        rng = _r.Random(42)
        for _ in range(20):
            samples.append({s: rng.uniform(0.6, 3.0) for s in syms})

    ratios = []
    for sub in samples:
        try:
            va = float(a.subs(sub))
            vb = float(b.subs(sub))
        except Exception:
            continue
        if not (math.isfinite(va) and math.isfinite(vb)):
            continue
        if abs(vb) < 1e-12:
            continue
        ratios.append(va / vb)

    if len(ratios) < 3:
        return False

    r0 = ratios[0]
    if abs(r0) < 1e-6:
        return False
    return all(abs(r - r0) <= atol * max(1.0, abs(r0)) for r in ratios)


@dataclass
class LoopState:
    iteration: int
    n_points: int
    n_candidates: int
    top_expr: str | None
    top_fit: float | None
    proposed_x: float | None
    converged: bool
    elapsed_s: float


@dataclass
class LoopResult:
    ground_truth: str
    converged: bool
    final_iteration: int
    final_n_points: int
    discovered_expr: str | None
    discovered_fit: float | None
    history: list[LoopState] = field(default_factory=list)


class ClosedLoopDiscovery:
    """Autonomous law discovery via propose → measure → refit loop.

    The ground_truth_expr is used ONLY for convergence verification.
    Discovery itself has no access to it — only to the simulated
    measurements it generates.
    """

    def __init__(
        self,
        ground_truth_expr: sp.Expr,
        x_range: tuple[float, float],
        *,
        var_name: str = "x",
        initial_n: int = 5,
        noise_frac: float = 0.02,
        max_iterations: int = 15,
        time_budget_direct: float = 5.0,
        time_budget_compose: float = 10.0,
        random_state: int = 0,
        verbose: bool = True,
    ):
        self.ground_truth = ground_truth_expr
        self.var_name = var_name
        self.x_range = x_range
        self.initial_n = initial_n
        self.noise_frac = noise_frac
        self.max_iterations = max_iterations
        self.time_budget_direct = time_budget_direct
        self.time_budget_compose = time_budget_compose
        self.rng = random.Random(random_state)
        self.verbose = verbose
        self._var = sp.Symbol(var_name)

    def _simulate(self, x: float) -> float:
        """Run the ground-truth experiment at x (with noise)."""
        try:
            y = float(self.ground_truth.subs({self._var: float(x)}))
        except Exception:
            return 0.0
        y *= (1.0 + self.rng.gauss(0.0, self.noise_frac))
        return y

    def _initial_data(self):
        """Generate initial_n random measurements."""
        data = []
        for _ in range(self.initial_n):
            x = self.rng.uniform(*self.x_range)
            data.append(({self.var_name: x}, self._simulate(x)))
        return data

    def _discover(self, data):
        try:
            return discover_and_classify(
                data, [self.var_name],
                time_budget_direct=self.time_budget_direct,
                time_budget_compose=self.time_budget_compose,
            )
        except Exception:
            return []

    def _check_converged(self, results, data=None):
        """Does any candidate match the ground truth up to scalar?

        Only checks top-10 candidates by fit_error to bound cost.
        """
        top = sorted(
            results, key=lambda r: r.get("fit_error", 1.0)
        )[:10]
        for r in top:
            if r.get("expr") is not None and _equivalent_up_to_scalar(
                r["expr"], self.ground_truth, data=data,
            ):
                return True, r
        return False, (results[0] if results else None)

    def run(self) -> LoopResult:
        t0 = time.time()
        data = self._initial_data()
        history: list[LoopState] = []
        converged = False
        final_expr = None
        final_fit = None

        for it in range(self.max_iterations):
            if self.verbose:
                print(f"\n  [iter {it:>2}] n={len(data)}  "
                      f"discovering...")

            results = self._discover(data)
            if not results:
                if self.verbose:
                    print(f"    no candidates  →  proposing random point")
                x_next = propose_experiment(
                    [], data, [self.var_name], self.x_range,
                    random_state=self.rng.randint(0, 10**9),
                )
                data.append(({self.var_name: x_next}, self._simulate(x_next)))
                history.append(LoopState(
                    iteration=it, n_points=len(data), n_candidates=0,
                    top_expr=None, top_fit=None,
                    proposed_x=x_next, converged=False,
                    elapsed_s=time.time() - t0,
                ))
                continue

            conv, top = self._check_converged(results, data)
            if top is not None:
                final_expr = str(top["expr"])
                final_fit = top["fit_error"]

            if self.verbose:
                tag = "✓" if conv else " "
                print(f"    {tag} top: [{top['class']:12s}] "
                      f"{str(top['expr'])[:45]:45s}  "
                      f"fit={top['fit_error']:.4f}  "
                      f"n_cand={len(results)}")

            if conv:
                converged = True
                history.append(LoopState(
                    iteration=it, n_points=len(data),
                    n_candidates=len(results),
                    top_expr=str(top["expr"]),
                    top_fit=top["fit_error"],
                    proposed_x=None, converged=True,
                    elapsed_s=time.time() - t0,
                ))
                if self.verbose:
                    print(f"    CONVERGED at iteration {it}")
                break

            x_next = propose_experiment(
                results, data, [self.var_name], self.x_range,
                random_state=self.rng.randint(0, 10**9),
            )
            y_next = self._simulate(x_next)
            data.append(({self.var_name: x_next}, y_next))

            if self.verbose:
                print(f"    → propose x={x_next:.4f}, measure y={y_next:.4f}")

            history.append(LoopState(
                iteration=it, n_points=len(data),
                n_candidates=len(results),
                top_expr=str(top["expr"]) if top else None,
                top_fit=top["fit_error"] if top else None,
                proposed_x=x_next, converged=False,
                elapsed_s=time.time() - t0,
            ))

        return LoopResult(
            ground_truth=str(self.ground_truth),
            converged=converged,
            final_iteration=len(history) - 1 if history else 0,
            final_n_points=len(data),
            discovered_expr=final_expr,
            discovered_fit=final_fit,
            history=history,
        )


__all__ = ["ClosedLoopDiscovery", "LoopResult", "LoopState"]
