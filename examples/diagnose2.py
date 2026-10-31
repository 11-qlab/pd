"""After routing fix, spot-check the 6 that failed. For each, call
run_one directly and print what came back."""
from purediscover.benchmark import run_one, _build_data, _grammar_for
from purediscover.resource.laws import get
from purediscover.rich_search import brute_search

REMAINING = [
    "pendulum_period",
    "stefan_boltzmann",
    "boltzmann_entropy",
    "ph",
    "hypotenuse",
    "compound_interest",
]

print("Spot-check of the 6 remaining failures (post-routing-fix):")
print("=" * 66)
for name in REMAINING:
    law = get(name)
    r = run_one(law)
    mark = "✓" if r["found"] else "✗"
    print(f"  {mark}  {name:22s} n={law.difficulty:<2} "
          f"{r['elapsed']:5.1f}s  "
          f"{r.get('found_expr', r.get('reason', ''))[:40]}")
    if not r["found"] and name in {"pendulum_period", "hypotenuse"}:
        # dig deeper on the two sqrt cases
        print(f"       target: {law.expr}")
        print(f"       ops_needed: {law.ops_needed}")
        g = _grammar_for(law)
        print(f"       grammar ops: {getattr(g, 'ops', '<hidden>')}")
        data = _build_data(law, n=40, seed=0)
        for max_n in (10, 12, 14):
            try:
                c = brute_search(g, data, max_nodes=max_n,
                                 max_per_level=50000, tol=0.02,
                                 time_budget=20)
                print(f"         max_nodes={max_n:>2}: {len(c)} candidates"
                      + (f"  best={c[0]['expr']}" if c else ""))
            except Exception as e:
                print(f"         max_nodes={max_n:>2}: raised {e}")
print("=" * 66)
