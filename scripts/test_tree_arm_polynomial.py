#!/usr/bin/env python3
import sys, time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
from purediscover_ext import tree_search_ext


def run_one(name, x, y, z, truth):
    data = [({"x": float(x[i]), "y": float(y[i])}, float(z[i]))
            for i in range(len(x))]
    print(f"\n=== {name} ===")
    print(f"  n={len(data)}  truth={truth}")
    t0 = time.time()
    cands = tree_search_ext(
        data, ["x", "y"],
        max_features=4, degree=2,
        tol=0.02, time_budget=15.0,
    )
    dt = time.time() - t0
    print(f"  took {dt:.2f}s, returned {len(cands)} candidates")
    for i, c in enumerate(cands[:5]):
        print(f"    [{i}] bits={c['bits']:8.2f}  mre={c['mre']:.3e}  "
              f"n={c['n_nodes']}  {c['expr']}")
    return cands


def main() -> int:
    rng = np.random.default_rng(42)
    n = 40
    x = rng.uniform(0.2, 2.0, n)
    y = rng.uniform(0.2, 2.0, n)

    z1 = 2.5*x + 3.0*y + 1.5 + rng.normal(0, 0.02, n)
    run_one("linear", x, y, z1, "2.5x+3y+1.5")

    z2 = 1.7*x*y + 0.3*x - 0.8*y + 2.0 + rng.normal(0, 0.02, n)
    run_one("bilinear", x, y, z2, "1.7xy+0.3x-0.8y+2")

    z3 = 1.5*x**2 + 2.0*y**2 - 1.0 + rng.normal(0, 0.02, n)
    run_one("quadratic", x, y, z3, "1.5x^2+2y^2-1")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
