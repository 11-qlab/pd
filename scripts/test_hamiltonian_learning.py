#!/usr/bin/env python3
"""Hamiltonian learning: recover J from <Z_1>(t) = cos(4*J*t) traces.

For each true J in {0.5, 1.0, 1.5, 2.0, 2.5}, generate 50 noisy points,
discover the frequency omega, report J_recovered = omega/4.
"""
import math, sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import re
from purediscover_quantum import make_trace_data
from purediscover_ext import discover_recursive


def main() -> int:
    true_Js = [0.5, 1.0, 1.5, 2.0, 2.5]
    print(f"{'J_true':>8s} {'J_recov':>10s} {'rel_err':>10s} {'mae':>10s} {'expr':<50s}")
    print("-" * 95)
    worst_rel = 0.0
    for J_true in true_Js:
        data = make_trace_data(J_true, n=50, t_periods=2.5,
                                noise_frac=0.02, seed=hash(J_true) & 0xFFFF)
        res = discover_recursive(data, ["t"], max_depth=2,
                                  noise_floor_factor=0.02,
                                  time_budget=30.0, verbose=False)
        expr = res["composed_expr"] or "?"
        mae = res["final_mae"] if res["final_mae"] is not None else float("inf")

        # extract frequency: match cos(NUM*t) or cos(NUM*t...)
        freq = None
        m = re.search(r"cos\(([0-9.]+)\*t", expr.replace(" ", ""))
        if m:
            try:
                freq = float(m.group(1))
            except ValueError:
                pass
        J_recov = (freq / 4.0) if freq is not None else float("nan")
        rel_err = abs(J_recov - J_true) / J_true if freq else float("nan")
        if freq:
            worst_rel = max(worst_rel, rel_err)

        print(f"{J_true:>8.3f} {J_recov:>10.4f} {rel_err:>10.2%} "
              f"{mae:>10.3e} {expr[:48]:<50s}")

    print()
    print(f"worst relative error across {len(true_Js)} systems: {worst_rel:.2%}")
    print(f"VERDICT: {'PASS' if worst_rel < 0.02 else 'REVIEW'}")
    return 0 if worst_rel < 0.02 else 1


if __name__ == "__main__":
    raise SystemExit(main())
