"""Stress test: does BOED help when initial sampling is insufficient?"""
from purediscover.benchmark.catalog import get_law
from purediscover.benchmark.runner import (
    run_random_growth, run_closed_loop_boed,
)

print("STRESS TEST: 2 initial points, 10% noise")
print("=" * 66)

for law_name in ("kepler_third", "arrhenius", "offset_inverse",
                 "stefan_boltzmann_T4", "linear"):
    law = get_law(law_name)
    print(f"\n### {law_name}")
    for method_name, fn in [("random_growth", run_random_growth),
                            ("closed_loop_boed", run_closed_loop_boed)]:
        try:
            r = fn(law, initial_n=2, max_iters=15, noise_frac=0.10,
                   budget=8.0)
            mark = "✓" if r.converged else "✗"
            print(f"  {mark} {method_name:20s}  "
                  f"iter={r.iterations:2d}  pts={r.points_used:2d}  "
                  f"time={r.time_s:5.1f}s")
        except Exception as e:
            print(f"  ! {method_name:20s}  {type(e).__name__}: {e}")
