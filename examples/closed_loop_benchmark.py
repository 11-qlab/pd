"""Closed-loop benchmark on 3 laws of increasing difficulty.

For each law:
    - start from 5 random measurements
    - run the propose → measure → refit loop
    - report iterations to converge

Laws:
    1. Kepler a^(3/2)   — power law, easy
    2. Linear 1.5*a     — linear, very easy
    3. Offset 2/a + 1   — offset form, uses offset peel
"""
import math
import time

import sympy as sp

from purediscover.closed_loop import ClosedLoopDiscovery


LAWS = [
    {
        "name": "Kepler's Third Law",
        "expr": sp.Symbol("a") ** sp.Rational(3, 2),
        "var": "a",
        "range": (0.5, 30.0),
        "initial_n": 5,
    },
    {
        "name": "Linear motion",
        "expr": sp.Float(1.5) * sp.Symbol("x"),
        "var": "x",
        "range": (0.5, 10.0),
        "initial_n": 5,
    },
    {
        "name": "Offset inverse",
        "expr": sp.Float(2.0) / sp.Symbol("x") + sp.Float(1.0),
        "var": "x",
        "range": (0.2, 20.0),
        "initial_n": 6,
    },
]


def main():
    print("CLOSED-LOOP BENCHMARK")
    print("=" * 78)
    print(f"  {len(LAWS)} laws, each started from ~5 random points")
    print("=" * 78)

    results = []
    for law in LAWS:
        print()
        print(f"\n### {law['name']}  (truth: {law['expr']})")
        print("-" * 78)

        loop = ClosedLoopDiscovery(
            ground_truth_expr=law["expr"],
            x_range=law["range"],
            var_name=law["var"],
            initial_n=law["initial_n"],
            noise_frac=0.02,
            max_iterations=12,
            time_budget_direct=6.0,
            time_budget_compose=12.0,
            random_state=0,
            verbose=True,
        )
        t0 = time.time()
        result = loop.run()
        elapsed = time.time() - t0

        results.append({
            "name": law["name"],
            "converged": result.converged,
            "iterations": result.final_iteration,
            "final_n": result.final_n_points,
            "discovered": result.discovered_expr,
            "elapsed": elapsed,
        })

    print()
    print("=" * 78)
    print("BENCHMARK SUMMARY")
    print("=" * 78)
    print(f"  {'law':28s}  {'converged':10s}  {'iter':4s}  {'pts':4s}  {'time':6s}")
    print("  " + "-" * 62)
    for r in results:
        mark = "✓" if r["converged"] else "✗"
        print(f"  {r['name']:28s}  {mark} {str(r['converged']):<9s}  "
              f"{r['iterations']:>3d}   {r['final_n']:>3d}   "
              f"{r['elapsed']:>5.1f}s")

    n_conv = sum(1 for r in results if r["converged"])
    print()
    print(f"  {n_conv}/{len(results)} laws converged to ground truth")


if __name__ == "__main__":
    main()
