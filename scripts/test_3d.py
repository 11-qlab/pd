#!/usr/bin/env python3
"""True 3D test: can search_templates_nd recover cos(2*x*y*z)?"""
import math, sys, time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
from purediscover_templates import search_templates_nd


def main() -> int:
    rng = np.random.default_rng(7)
    n = 60
    x = rng.uniform(0.1, 1.5, n)
    y = rng.uniform(0.1, 1.5, n)
    z = rng.uniform(0.1, 1.5, n)
    vals = 2.0 * x * y * z
    target = np.cos(vals) + rng.normal(0, 0.02, n)

    data = [({"x": float(x[i]), "y": float(y[i]), "z": float(z[i])},
             float(target[i])) for i in range(n)]
    print(f"n={n}   truth: y = cos(2*x*y*z)   noise=2%")
    print(f"x,y,z ranges [0.1, 1.5]^3")
    print()

    t0 = time.time()
    results = search_templates_nd(
        data, ["x", "y", "z"],
        max_order=3, max_mre=0.15,
        n_multistart=15, seed=42,
    )
    dt = time.time() - t0
    print(f"{len(results)} candidates fitted in {dt:.1f}s")
    print()
    for i, r in enumerate(results[:8]):
        print(f"  [{i}] composite={r.composite:15s}  "
              f"mae={r.mae:.3e}  bits={r.mdl_bits:8.1f}  ok={r.ok}")
        print(f"       {r.expr_str[:100]}")
    print()

    if results:
        w = results[0]
        has_cos = "cos" in w.expr_str
        # Accept sin(x + π/2) as equivalent to cos(x):
        # params[2] is the phase for a*x+b*x+c style templates
        has_cos_equiv = False
        for p in (w.params or []):
            try:
                if abs(abs(float(p)) - math.pi/2) < 0.15:
                    has_cos_equiv = "sin" in w.expr_str
                    break
            except (TypeError, ValueError):
                continue
        print(f"=== WINNER ===")
        print(f"  {w.expr_str[:120]}")
        print(f"  mae = {w.mae:.3e}   bits = {w.mdl_bits:.1f}")
        print()
        print(f"  contains cos        : {has_cos}")
        print(f"  contains sin(x+π/2) : {has_cos_equiv}")
        print(f"  composite used      : {w.composite}")
        print(f"  target composite    : x*y*z")
        print(f"  composite match     : {w.composite in ['x*y*z', 'y*x*z', 'z*x*y']}")
        print(f"  mae < 0.05          : {w.mae < 0.05}")

        # qualitative: is the composite x*y*z?
        ok = (has_cos or has_cos_equiv) and w.mae < 0.05 and \
             "x" in w.composite and "y" in w.composite and "z" in w.composite
        print()
        print(f"  VERDICT: {'PASS' if ok else 'REVIEW'}")
        return 0 if ok else 1
    print("FAIL: no candidates")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
