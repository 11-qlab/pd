"""Benchmark framework for law discovery methods.

Public API:

    from purediscover.benchmark import (
        Law, LawCatalog, make_data,
        SingleShot, RandomGrowth, ClosedLoopBOED,
        Benchmark, ReportRenderer,
        LawResult, MethodSummary, BenchmarkResult,
    )

    catalog = LawCatalog.default()
    benchmark = Benchmark(
        catalog=catalog,
        methods=[SingleShot(), RandomGrowth(), ClosedLoopBOED()],
    )
    result = benchmark.run(noise_frac=0.02)
    print(ReportRenderer(result).render_full())
    result.to_json("benchmark.json")
"""
from purediscover.benchmark.catalog import (
    CATALOG, Law, LawCatalog, by_difficulty, get_law, make_data,
)
from purediscover.benchmark.methods import (
    ClosedLoopBOED, DiscoveryMethod, RandomGrowth, SingleShot,
)
from purediscover.benchmark.results import (
    BenchmarkResult, LawResult, MethodSummary,
)
from purediscover.benchmark.runner import Benchmark
from purediscover.benchmark.report import ReportRenderer
from purediscover.benchmark.convergence import ConvergenceChecker
from purediscover.benchmark.phase_catalog import (
    PhaseCatalog, build_phase_laws, full_catalog,
)

__all__ = [
    "Law", "LawCatalog", "CATALOG", "make_data",
    "get_law", "by_difficulty",
    "DiscoveryMethod", "SingleShot", "RandomGrowth", "ClosedLoopBOED",
    "Benchmark", "ReportRenderer", "ConvergenceChecker",
    "PhaseCatalog", "build_phase_laws", "full_catalog",
    "LawResult", "MethodSummary", "BenchmarkResult",
]
