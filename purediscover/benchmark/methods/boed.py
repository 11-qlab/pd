"""Closed-loop BOED method: single-variable only."""
from __future__ import annotations

import time

from purediscover.benchmark.catalog import Law
from purediscover.benchmark.methods.base import DiscoveryMethod
from purediscover.benchmark.results import LawResult
from purediscover.closed_loop import ClosedLoopDiscovery


class ClosedLoopBOED(DiscoveryMethod):
    name = "closed_loop_boed"

    def __init__(self, initial_n: int = 5, max_iters: int = 10,
                 budget: float = 8.0):
        self.initial_n = initial_n
        self.max_iters = max_iters
        self.budget = budget

    def supports(self, law: Law) -> bool:
        return len(law.var_names) == 1

    def run(self, law: Law, *, noise_frac: float = 0.02, **_) -> LawResult:
        if not self.supports(law):
            return LawResult(
                law_name=law.name, method=self.name,
                converged=False, iterations=0, points_used=0,
                time_s=0.0, skipped=True,
                error="BOED requires single-variable data",
            )

        var_name = law.var_names[0]
        lo, hi = law.var_ranges[var_name]
        loop = ClosedLoopDiscovery(
            ground_truth_expr=law.expr,
            x_range=(lo, hi),
            var_name=var_name,
            initial_n=self.initial_n,
            noise_frac=noise_frac,
            max_iterations=self.max_iters,
            time_budget_direct=self.budget / 2,
            time_budget_compose=self.budget / 2,
            random_state=law.seed,
            verbose=False,
        )
        t0 = time.time()
        try:
            result = loop.run()
        except Exception as e:
            return LawResult(
                law_name=law.name, method=self.name,
                converged=False, iterations=0, points_used=0,
                time_s=time.time() - t0,
                error=f"{type(e).__name__}: {e}",
            )

        return LawResult(
            law_name=law.name, method=self.name,
            converged=result.converged,
            iterations=result.final_iteration,
            points_used=result.final_n_points,
            time_s=time.time() - t0,
            final_expr=result.discovered_expr,
            novelty_class=None,
        )
