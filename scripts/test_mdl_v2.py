"""Compare mdl_score (v1) vs mdl_score_v2 on canonical cases."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import sympy as sp
from purediscover.mdl import mdl_score, mdl_score_v2


def compare(exprs, data, label, tol=0.02):
    print(f"\n=== {label} ===")
    print(f"{'expression':<42} {'v1 bits':>10} {'v2 bits':>10} {'v1 rk':>6} {'v2 rk':>6}")
    print("-" * 80)

    rows = []
    for e in exprs:
        r1 = mdl_score(e, data, residual_tol=tol)
        r2 = mdl_score_v2(e, data, residual_tol=tol)
        rows.append((e, r1, r2))

    ok1 = sorted([(i, r1) for i, (_, r1, _) in enumerate(rows) if r1["ok"]],
                 key=lambda x: x[1]["bits"])
    ok2 = sorted([(i, r2) for i, (_, _, r2) in enumerate(rows) if r2["ok"]],
                 key=lambda x: x[1]["bits"])
    rk1 = {i: k + 1 for k, (i, _) in enumerate(ok1)}
    rk2 = {i: k + 1 for k, (i, _) in enumerate(ok2)}

    for i, (e, r1, r2) in enumerate(rows):
        b1 = f"{r1['bits']:.1f}" if r1["ok"] else "fail"
        b2 = f"{r2['bits']:.1f}" if r2["ok"] else "fail"
        print(f"{str(e):<42} {b1:>10} {b2:>10} "
              f"{str(rk1.get(i, '-')):>6} {str(rk2.get(i, '-')):>6}")

    best1 = str(rows[ok1[0][0]][0]) if ok1 else "(none)"
    best2 = str(rows[ok2[0][0]][0]) if ok2 else "(none)"
    if not ok1 and not ok2:
        agree = "BOTH REJECT (no evidence)"
    elif best1 == best2:
        agree = "AGREE"
    else:
        agree = "DIFFER"
    print(f"  best v1: {best1}")
    print(f"  best v2: {best2}   [{agree}]")
    return (bool(ok1) and bool(ok2) and best1 == best2), agree


def main():
    x = sp.Symbol("x")
    t = np.linspace(0, 10, 60)

    # Case 1: noiseless linear
    data1 = [(dict(x=float(v)), 3.0 * v + 1.0) for v in t]
    e1 = [sp.Integer(1) + 3 * x,
          sp.Integer(1) + 3.1 * x,
          sp.Integer(1) + 3 * x + 0.1 * x**2,
          x,
          sp.Integer(5)]
    a1, lab1 = compare(e1, data1, "Linear y = 3x + 1 (noiseless)")

    # Case 2: noiseless cosine
    data2 = [(dict(x=float(v)), float(np.cos(2 * v))) for v in t]
    e2 = [sp.cos(2 * x),
          sp.cos(2.01 * x),
          sp.cos(x),
          sp.cos(2 * x) + 0.01 * sp.sin(0.5 * x),
          sp.Integer(0)]
    a2, lab2 = compare(e2, data2, "Cosine y = cos(2x) (noiseless)")

    # Case 3: cosine with noise
    rng = np.random.default_rng(0)
    data3 = [(dict(x=float(v)), float(np.cos(2 * v) + 0.02 * rng.standard_normal()))
             for v in t]
    e3 = [sp.cos(2 * x),
          sp.cos(2.01 * x),
          sp.cos(x),
          sp.cos(2 * x) + 0.02 * sp.sin(2 * x)]
    # tol=0.1 because 2% additive noise on |y|=1 gives max residual ~5%,
    # which correctly exceeds tol=0.02. This case tests ranking among
    # near-fit candidates, not strict fit acceptance.
    a3, lab3 = compare(e3, data3, "Cosine y = cos(2x) + 2% noise", tol=0.1)

    # Case 4: damped Rabi
    data4 = [(dict(x=float(v)), float(np.cos(2 * v) * np.exp(-v / 2))) for v in t]
    e4 = [sp.cos(2 * x) * sp.exp(-x / 2),
          sp.cos(2 * x),
          sp.cos(2 * x) * sp.exp(-x / 2.05)]
    a4, lab4 = compare(e4, data4, "Damped cos(2x)*exp(-x/2)")

    print("\n" + "=" * 60)
    for i, (ok, lab) in enumerate([(a1, lab1), (a2, lab2), (a3, lab3), (a4, lab4)], 1):
        print(f"  case {i}: {lab}")
    valid = sum([a1, a2, a3, a4])
    print(f"\nValid agreement: {valid}/4")
    return 0 if valid >= 3 else 1


if __name__ == "__main__":
    sys.exit(main())
