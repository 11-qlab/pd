#!/usr/bin/env python3
"""1D additive test: y = cos(2*t) + 0.5*sin(0.5*t).
No single template fits both frequencies. Recursion must decompose.
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
from purediscover_ext import discover_recursive


def main() -> int:
    rng = np.random.default_rng(7)
    n = 100
    t = rng.uniform(0.1, 20.0, n)   # ~3 full cycles of 0.5-rad component
    y_true = np.cos(2.0 * t) + 0.5 * np.sin(0.5 * t)
    y_noisy = y_true + rng.normal(0, 0.02, n)

    data = [({"t": float(t[i])}, float(y_noisy[i])) for i in range(n)]
    print(f"n={n}   truth: y = cos(2*t) + 0.5*sin(0.5*t)   noise sigma=0.02")
    print()

    result = discover_recursive(
        data, ["t"], max_depth=2,
        noise_floor_factor=0.02,
        time_budget=30.0, verbose=True,
    )

    print()
    print("=== result ===")
    print(f"  depth            : {result['depth']}")
    print(f"  composition_mode : {result['composition_mode']}")
    print(f"  composed_expr    : {result['composed_expr']}")
    print(f"  final_mae        : {result['final_mae']}")
    expr = str(result["composed_expr"]).replace(" ", "")
    has_two_freqs = (expr.count("cos") + expr.count("sin")) >= 2
    is_additive = result["composition_mode"] == "additive"
    print()
    print(f"  two frequencies found : {has_two_freqs}")
    print(f"  additive composition  : {is_additive}")
    ok = has_two_freqs and is_additive
    print()
    print(f"  VERDICT: {'PASS' if ok else 'REVIEW'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
