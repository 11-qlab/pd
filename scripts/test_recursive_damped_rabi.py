#!/usr/bin/env python3
"""Damped Rabi: y = cos(2*Bx*t) * exp(-t/T2).
Expect: recursive discovery recovers cos(2*Bx*t) first, then exp(-t/T2)."""
import math, sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
from purediscover_ext import discover_recursive


def main() -> int:
    rng = np.random.default_rng(42)
    T2 = 2.0
    n = 80

    Bx = rng.uniform(0.3, 1.8, n)
    t = rng.uniform(0.1, 3.0, n)
    y_true = np.cos(2.0 * Bx * t) * np.exp(-t / T2)
    y_noisy = y_true * (1.0 + rng.normal(0, 0.02, n))

    data = [({"Bx": float(Bx[i]), "t": float(t[i])}, float(y_noisy[i]))
            for i in range(n)]
    print(f"n={n}   truth: y = cos(2*Bx*t) * exp(-t/{T2})   noise=2%")
    print()

    result = discover_recursive(data, ["Bx", "t"],
                                max_depth=2,
                                noise_floor_factor=0.02,
                                time_budget=40.0,
                                verbose=True)

    print()
    print("=== result ===")
    print(f"  depth            : {result['depth']}")
    print(f"  composition_mode : {result['composition_mode']}")
    print(f"  composed_expr    : {result['composed_expr']}")
    print(f"  final_mae        : {result['final_mae']}")

    # check for expected structure
    expr = str(result["composed_expr"]).replace(" ", "")
    has_cos = "cos" in expr
    has_exp = "exp" in expr
    print()
    print(f"  contains cos     : {has_cos}")
    print(f"  contains exp     : {has_exp}")
    print(f"  multiplicative   : {result['composition_mode'] == 'multiplicative'}")

    # Accept EITHER:
    #  - multiplicative composition (recursive path found it in pieces), OR
    #  - single composition with both cos and exp in one expression
    #    (template captured the whole law directly)
    mode_ok = (result["composition_mode"] == "multiplicative"
               or (result["composition_mode"] == "single" and has_exp))
    ok = has_cos and has_exp and mode_ok
    print()
    print(f"  VERDICT: {'PASS' if ok else 'REVIEW'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
