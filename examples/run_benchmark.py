"""Benchmark CLI.

    python examples/run_benchmark.py                       # quick (3 laws)
    python examples/run_benchmark.py --full                # all laws
    python examples/run_benchmark.py --full --latex        # paper-ready table
    python examples/run_benchmark.py --laws kepler_third,ohm
    python examples/run_benchmark.py --methods single_shot,random_growth
"""
import argparse
import pathlib
import sys

from purediscover.benchmark import (
    Benchmark, ClosedLoopBOED, LawCatalog, PhaseCatalog, RandomGrowth,
    ReportRenderer, SingleShot, full_catalog,
)


QUICK_LAWS = ["kepler_third", "ohm", "arrhenius"]

METHOD_REGISTRY = {
    "single_shot": SingleShot,
    "random_growth": RandomGrowth,
    "closed_loop_boed": ClosedLoopBOED,
}


def _pick_laws(args):
    catalog = LawCatalog.default()
    if args.laws:
        return args.laws.split(",")
    if args.full:
        return None  # all
    return QUICK_LAWS


def _pick_methods(args):
    if args.methods == "all":
        return list(METHOD_REGISTRY.keys())
    return args.methods.split(",")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--laws", default="")
    ap.add_argument("--methods", default="all")
    ap.add_argument("--noise", type=float, default=0.02)
    ap.add_argument("--output", default="benchmark_results.json")
    ap.add_argument("--domain",
                    choices=["classical", "interferometric", "all"],
                    default="all",
                    help="which catalog(s) to run")
    ap.add_argument("--format", choices=["text", "markdown", "latex", "json"],
                    default="text")
    ap.add_argument("--markdown", action="store_true")
    ap.add_argument("--latex", action="store_true")
    args = ap.parse_args()

    if args.markdown:
        args.format = "markdown"
    if args.latex:
        args.format = "latex"

    laws = _pick_laws(args)
    method_names = _pick_methods(args)

    for m in method_names:
        if m not in METHOD_REGISTRY:
            print(f"unknown method: {m}", file=sys.stderr)
            sys.exit(1)

    if args.domain == "classical":
        catalog = LawCatalog.default()
    elif args.domain == "interferometric":
        catalog = PhaseCatalog.default()
    else:
        catalog = full_catalog()
    if laws:
        catalog = catalog.filter(names=laws)

    methods = [METHOD_REGISTRY[m]() for m in method_names]

    print("BENCHMARK RUN")
    print("=" * 100)
    print(f"  laws    : {len(catalog)}  "
          f"({', '.join(catalog.names()[:3])}"
          + ("..." if len(catalog) > 3 else "") + ")")
    print(f"  methods : {method_names}")
    print(f"  noise   : {args.noise * 100:.0f}%")
    print("=" * 100)

    def _progress(law, method):
        print(f"  running {law} / {method}...", flush=True)

    benchmark = Benchmark(catalog=catalog, methods=methods)
    result = benchmark.run(noise_frac=args.noise, progress=_progress)

    renderer = ReportRenderer(result)
    print()
    if args.format == "text":
        print(renderer.render_full())
    elif args.format == "markdown":
        print(renderer.render_markdown())
    elif args.format == "latex":
        print(renderer.render_latex())

    result.to_json(args.output)
    print(f"\n  wrote {args.output}")


if __name__ == "__main__":
    main()
