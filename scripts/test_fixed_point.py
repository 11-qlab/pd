import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from purediscover_ext.fixed_point import (
    generate_iteration_traces, discover_update_rule,
    colebrook_solve, solve_with_discovered,
)

def main():
    print("=" * 60)
    print("Fixed-point discovery (Colebrook)")
    print("=" * 60)

    train = generate_iteration_traces(n_pts=1000, seed=0)
    rule = discover_update_rule(train)
    print(f"Discovery path: {rule['kind']}")
    if rule["kind"] == "parametric":
        for k, v in rule["params"].items():
            print(f"  {k} = {v:+.6f}   (truth: {'-2' if k=='a' else '1/3.7' if k=='b' else '2.51' if k=='c' else '0'})")
        print(f"  training residual RMS = {rule['residual_rms']:.3e}")

    rng = np.random.default_rng(1)
    n = 200
    Re_t = 10 ** rng.uniform(4, 7, n)
    ks_t = 10 ** rng.uniform(-5, -1.5, n)
    oracle = np.array([colebrook_solve(Re_t[i], ks_t[i]) for i in range(n)])

    if rule["kind"] == "parametric":
        pred = np.array([solve_with_discovered(rule, Re_t[i], ks_t[i]) for i in range(n)])
        rel = np.abs(pred - oracle) / oracle
        print(f"\nHeld-out fixed-point convergence:")
        print(f"  max rel err  = {rel.max():.3e}")
        print(f"  mean rel err = {rel.mean():.3e}")
        ok = rel.max() < 1e-3
        print(f"  PASS (max < 1e-3): {ok}")
        return 0 if ok else 1
    else:
        print("\nSymbolic path active — evaluate composed expression on held-out data.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
