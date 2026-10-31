"""Closed-loop discovery demo: recover Kepler's Third Law.

Starts with 5 random measurements of a^(3/2). Runs the propose → measure
→ refit loop until discovery converges to the true form (or budget
exhausts).

Expected: converge in 1-3 iterations (Kepler is a simple power law).
"""
import sympy as sp

from purediscover.closed_loop import ClosedLoopDiscovery


def main():
    a = sp.Symbol("a")
    ground_truth = a ** sp.Rational(3, 2)

    print("CLOSED-LOOP DISCOVERY — Kepler's Third Law")
    print("=" * 66)
    print(f"  Ground truth: {ground_truth}")
    print(f"  Range       : a ∈ [0.5, 30]")
    print(f"  Initial N   : 5 random points")
    print(f"  Noise       : 2%")
    print(f"  Max iters   : 15")
    print("=" * 66)

    loop = ClosedLoopDiscovery(
        ground_truth_expr=ground_truth,
        x_range=(0.5, 30.0),
        var_name="a",
        initial_n=5,
        noise_frac=0.02,
        max_iterations=15,
        time_budget_direct=5.0,
        time_budget_compose=10.0,
        random_state=42,
        verbose=True,
    )

    result = loop.run()

    print()
    print("=" * 66)
    print(f"  Converged       : {result.converged}")
    print(f"  Final iteration : {result.final_iteration}")
    print(f"  Total points    : {result.final_n_points}")
    print(f"  Discovered form : {result.discovered_expr}")
    print(f"  Ground truth    : {result.ground_truth}")
    print("=" * 66)


if __name__ == "__main__":
    main()
