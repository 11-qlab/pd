#!/usr/bin/env python3
"""4D verification: recover cos(2*w*x*y*z) from 2% noise.
Tests composite enumeration at N=4 (was tested only up to N=3).
"""
import math, sys, time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
from purediscover_templates import search_templates_nd


def main() -> int:
    rng = np.random.default_rng(11)
    n = 80
    w = rng.uniform(0.1, 1.2, n)
    x = rng.uniform(0.1, 1.2, n)
    y = rng.uniform(0.1, 1.2, n)
    z = rng.uniform(0.1, 1.2, n)
    target = np.cos(2.0 * w * x * y * z) + rng.normal(0, 0.02, n)

    data = [({"w": float(w[i]), "x": float(x[i]),
             "y": float(y[i]), "z": float(z[i])}, float(target[i]))
            for i in range(n)]
    print(f"n={n}   truth: y = cos(2*w*x*y*z)   noise=2%")
    print(f"w,x,y,z ∈ [0.1, 1.2]^4")
    print()

    t0 = time.time()
    results = search_templates_nd(
        data, ["w", "x", "y", "z"],
        max_order=4, max_mre=0.15,
        n_multistart=10, seed=42,
    )
    dt = time.time() - t0
    print(f"{len(results)} candidates fitted in {dt:.1f}s")
    print()
    for i, r in enumerate(results[:5]):
        print(f"  [{i}] composite={r.composite:20s}  "
              f"mae={r.mae:.3e}  bits={r.mdl_bits:8.1f}  ok={r.ok}")
        print(f"       {r.expr_str[:100]}")
    print()

    if not results:
        print("VERDICT: FAIL (no candidates)")
        return 1

    w_best = results[0]
    has_cos = "cos" in w_best.expr_str
    has_sin_pi2 = False
    if "sin" in w_best.expr_str:
        for p in (w_best.params or []):
            try:
                if abs(abs(float(p)) - math.pi/2) < 0.15:
                    has_sin_pi2 = True
                    break
            except (TypeError, ValueError):
                continue
    composite_ok = all(v in w_best.composite for v in ["w", "x", "y", "z"])
    mae_ok = w_best.mae < 0.05

    print(f"  cos or sin(θ+π/2)   : {has_cos or has_sin_pi2}")
    print(f"  composite has all 4 : {composite_ok}")
    print(f"  mae < 0.05          : {mae_ok}")
    ok = (has_cos or has_sin_pi2) and composite_ok and mae_ok
    print()
    print(f"  VERDICT: {'PASS' if ok else 'REVIEW'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
