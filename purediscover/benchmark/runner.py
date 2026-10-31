"""Benchmark orchestrator."""
from __future__ import annotations

import statistics
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Callable

from purediscover.benchmark.catalog import Law, LawCatalog
from purediscover.benchmark.methods.base import DiscoveryMethod
from purediscover.benchmark.results import (
    BenchmarkResult, LawResult, MethodSummary,
)


class Benchmark:
    def __init__(self, catalog: LawCatalog, methods: list[DiscoveryMethod]):
        self.catalog = catalog
        self.methods = methods

    def run(self, *,
            noise_frac: float = 0.02,
            law_names: list[str] | None = None,
            method_names: list[str] | None = None,
            progress: Callable[[str, str], None] | None = None,
            ) -> BenchmarkResult:
        laws = (self.catalog.filter(names=law_names)
                if law_names else self.catalog)
        methods = (
            [m for m in self.methods if m.name in method_names]
            if method_names else self.methods
        )

        if progress is None:
            progress = lambda law, method: None

        t_start = time.time()
        results: list[LawResult] = []
        for law in laws:
            for method in methods:
                progress(law.name, method.name)
                try:
                    r = method.run(law, noise_frac=noise_frac)
                except Exception as e:
                    r = LawResult(
                        law_name=law.name, method=method.name,
                        converged=False, iterations=0, points_used=0,
                        time_s=0.0,
                        error=f"{type(e).__name__}: {e}",
                    )
                results.append(r)

        wall = time.time() - t_start
        summaries = _summarize(results, method_names or [m.name for m in methods])

        return BenchmarkResult(
            timestamp=datetime.now(timezone.utc).isoformat(),
            noise_frac=noise_frac,
            wall_time_s=wall,
            n_laws=len(list(laws)),
            law_names=[l.name for l in laws],
            method_names=[m.name for m in methods],
            results=results,
            summaries=summaries,
        )


def _summarize(results: list[LawResult],
               method_order: list[str]) -> list[MethodSummary]:
    by_method: dict[str, list[LawResult]] = defaultdict(list)
    for r in results:
        by_method[r.method].append(r)

    summaries: list[MethodSummary] = []
    for method in method_order:
        rs = by_method.get(method, [])
        applicable = [r for r in rs if not r.skipped]
        converged = [r for r in applicable if r.converged]
        summaries.append(MethodSummary(
            method=method,
            n_total=len(rs),
            n_applicable=len(applicable),
            n_skipped=len(rs) - len(applicable),
            n_converged=len(converged),
            convergence_rate=(len(converged) / len(applicable)
                              if applicable else 0.0),
            median_time_s=_median([r.time_s for r in applicable]),
            median_iterations=_median([r.iterations for r in converged]),
            median_points=_median([r.points_used for r in converged]),
        ))
    return summaries


def _median(xs: list[float | int]) -> float | None:
    if not xs:
        return None
    return float(statistics.median(xs))


__all__ = ["Benchmark"]
