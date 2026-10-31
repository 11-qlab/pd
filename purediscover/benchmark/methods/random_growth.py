"""Random growth: add one random measurement per iteration."""
from __future__ import annotations

import random
import time

import sympy as sp

from purediscover.benchmark.catalog import Law, make_data
from purediscover.benchmark.convergence import ConvergenceChecker
from purediscover.benchmark.methods.base import DiscoveryMethod
from purediscover.benchmark.results import LawResult
from purediscover.discover import discover_and_classify


class RandomGrowth(DiscoveryMethod):
    name = "random_growth"

    def __init__(self, initial_n: int = 5, max_iters: int = 10,
                 budget: float = 6.0,
                 convergence: ConvergenceChecker | None = None):
        self.initial_n = initial_n
        self.max_iters = max_iters
        self.budget = budget
        self.checker = convergence or ConvergenceChecker()

    def run(self, law: Law, *, noise_frac: float = 0.02, **_) -> LawResult:
        rng = random.Random(law.seed)
        data = make_data(law, self.initial_n, noise_frac)
        t0 = time.time()

        def _grow_one():
            env = {}
            for name in law.var_names:
                lo, hi = law.var_ranges[name]
                env[name] = rng.uniform(lo, hi)
            try:
                y = float(law.expr.subs(
                    {sp.Symbol(k): v for k, v in env.items()}
                ))
            except Exception:
                return None
            y *= (1.0 + rng.gauss(0.0, noise_frac))
            return env, y

        results: list[dict] = []
        for it in range(self.max_iters + 1):
            try:
                results = discover_and_classify(
                    data, law.var_names,
                    time_budget_direct=self.budget / 3,
                    time_budget_compose=self.budget / 3,
                )
            except Exception as e:
                return LawResult(
                    law_name=law.name, method=self.name,
                    converged=False, iterations=it,
                    points_used=len(data), time_s=time.time() - t0,
                    error=f"{type(e).__name__}: {e}",
                )

            top_candidates = sorted(
                results, key=lambda r: r.get("fit_error", 1.0)
            )[:10]
            for r in top_candidates:
                if self.checker.is_converged(r["expr"], law.expr, data):
                    return LawResult(
                        law_name=law.name, method=self.name,
                        converged=True, iterations=it,
                        points_used=len(data),
                        time_s=time.time() - t0,
                        final_expr=str(r["expr"]),
                        novelty_class=r["class"],
                    )

            if it == self.max_iters:
                break
            sample = _grow_one()
            if sample is not None:
                data.append(sample)

        return LawResult(
            law_name=law.name, method=self.name,
            converged=False, iterations=self.max_iters,
            points_used=len(data), time_s=time.time() - t0,
            final_expr=str(results[0]["expr"]) if results else None,
            novelty_class=results[0]["class"] if results else None,
        )
