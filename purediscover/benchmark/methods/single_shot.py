"""Single-shot method: fixed dataset, one discover call."""
from __future__ import annotations

import time

from purediscover.benchmark.catalog import Law, make_data
from purediscover.benchmark.convergence import ConvergenceChecker
from purediscover.benchmark.methods.base import DiscoveryMethod
from purediscover.benchmark.results import LawResult
from purediscover.discover import discover_and_classify


class SingleShot(DiscoveryMethod):
    name = "single_shot"

    def __init__(self, n_points: int = 15, budget: float = 8.0,
                 convergence: ConvergenceChecker | None = None):
        self.n_points = n_points
        self.budget = budget
        self.checker = convergence or ConvergenceChecker()

    def run(self, law: Law, *, noise_frac: float = 0.02, **_) -> LawResult:
        data = make_data(law, self.n_points, noise_frac)
        t0 = time.time()
        try:
            results = discover_and_classify(
                data, law.var_names,
                time_budget_direct=self.budget / 2,
                time_budget_compose=self.budget / 2,
            )
        except Exception as e:
            return LawResult(
                law_name=law.name, method=self.name,
                converged=False, iterations=0, points_used=len(data),
                time_s=time.time() - t0,
                error=f"{type(e).__name__}: {e}",
            )

        elapsed = time.time() - t0
        # Only check top 10 candidates (by fit_error) to avoid running
        # symbolic convergence on hundreds of low-quality fits
        top_candidates = sorted(
            results, key=lambda r: r.get("fit_error", 1.0)
        )[:10]
        for r in top_candidates:
            if self.checker.is_converged(r["expr"], law.expr, data):
                return LawResult(
                    law_name=law.name, method=self.name,
                    converged=True, iterations=0, points_used=len(data),
                    time_s=elapsed, final_expr=str(r["expr"]),
                    novelty_class=r["class"],
                )

        return LawResult(
            law_name=law.name, method=self.name,
            converged=False, iterations=0, points_used=len(data),
            time_s=elapsed,
            final_expr=str(results[0]["expr"]) if results else None,
            novelty_class=results[0]["class"] if results else None,
        )
