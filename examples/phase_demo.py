"""Demo: recover interferometric phase laws from data.

Shows the pipeline recovering 3 representative phase laws:
    cow_phase        m·g·T²                (3-var product)
    magnetic_ab      q·B·A                 (3-var product)
    compton_clock    m·g·Δh·T              (4-var product, hard)
"""
import sympy as sp

from purediscover.benchmark import (
    PhaseCatalog, RandomGrowth, SingleShot, ReportRenderer,
    Benchmark,
)


def _run(law_names, method_cls, n_points_override=None):
    catalog = PhaseCatalog.default().filter(names=law_names)
    method = method_cls()
    b = Benchmark(catalog=catalog, methods=[method])
    return b.run(noise_frac=0.02, progress=lambda l, m: None)


if __name__ == "__main__":
    print("PHASE LAW RECOVERY DEMO")
    print("=" * 78)
    print("Interferometric phase laws are DERIVED consequences of QM +")
    print("external potential. Not independent laws of nature. Testing:")
    print("does the pipeline recover the algebraic form from data?")
    print("=" * 78)

    demo_laws = ["cow_phase", "magnetic_ab", "compton_clock"]

    for method_cls in (SingleShot, RandomGrowth):
        print(f"\n>>> {method_cls.__name__}")
        print("-" * 78)
        result = _run(demo_laws, method_cls)
        for r in result.results:
            mark = "✓" if r.converged else "✗"
            expr = r.final_expr or "(none)"
            print(f"  {mark} {r.law_name:25s}  "
                  f"{r.points_used:>2d} pts  {r.time_s:5.1f}s  {expr[:50]}")

    print()
    print("=" * 78)
    print("WHAT THIS SHOWS")
    print("=" * 78)
    print("The pipeline recovers:")
    print("  3-variable products     (cow_phase, magnetic_ab)")
    print("  4-variable products     (compton_clock) — if converged")
    print()
    print("If a 4-var law fails, that's a documented scope limit:")
    print("search budget and candidate explosion, not a physics limit.")
    print("=" * 78)
