"""Force the pipeline to fit pure noise, then watch the gate reject it."""
import random
import sympy as sp

from purediscover.discover import discover_and_classify
from purediscover.novelty_gate import full_gate, render_report


rng = random.Random(11)
data = [({"x": rng.uniform(0.1, 10.0)}, rng.uniform(0.0, 1.0))
        for _ in range(30)]

print(f"Generated {len(data)} random (x, y) points")
print("Running discover_and_classify with LOOSE tol=0.20...")
print()

results = discover_and_classify(
    data, ["x"],
    time_budget_direct=30,
    time_budget_compose=60,
    tol=0.20,   # ← loosened so the search fits SOMETHING
)

if not results:
    print("(still nothing — the search is genuinely incapable on this data)")
    print("That's a stronger rejection than the gate-level case.")
else:
    top = results[0]
    print(f"top candidate: [{top['class']}] {top['expr']}")
    print(f"fit_error = {top['fit_error']:.4f}, n_nodes = {top['n_nodes']}")
    print()

    gate = full_gate(top["expr"], data, ["x"], n_bootstrap=30)
    print(render_report(gate))
