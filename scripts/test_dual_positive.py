#!/usr/bin/env python3
"""Positive control for dual search: y = a^2 + b^2 should reduce in square-frame."""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
from purediscover_dual import search_dual


def main() -> int:
    rng = np.random.default_rng(0)
    n = 400
    a = rng.uniform(0.5, 3.0, n)
    b = rng.uniform(0.5, 3.0, n)
    y = a ** 2 + b ** 2
    data = [({"a": float(a[i]), "b": float(b[i])}, float(y[i])) for i in range(n)]

    print("=== positive-control: y = a^2 + b^2 ===")
    print("expected dual: u=a^2, v=b^2  ->  y = u + v  (much simpler)")
    print()
    result = search_dual(
        data, ["a", "b"],
        min_improvement=1.30,
        time_budget_direct=10.0,
        time_budget_compose=20.0,
        tol=0.05,
    )
    print()
    print(f"baseline_law : {result['baseline_law']}")
    print(f"baseline_mdl : {result['baseline_mdl']:.2f}")
    print(f"attempted    : {result['attempted']}  valid: {result['valid']}")
    if result["dual"]:
        d = result["dual"]
        print(f"DUAL FOUND   : transforms={d.transforms}")
        print(f"               law={d.law_str}")
        print(f"               mdl={d.mdl:.2f}  improvement={d.improvement:.2f}x")
        return 0
    print("NO DUAL FOUND — positive control FAILED")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
