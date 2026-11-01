#!/usr/bin/env python3
"""Smoke-test module (b) dual + module (c) quantum oracle.

Self-locating: inserts the project root (parent of this script's dir)
at sys.path[0] so imports work regardless of cwd or invocation path.
"""
import math
import sys
from pathlib import Path

# --- self-locate: project root = parent of scripts/ -----------------
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
# -------------------------------------------------------------------

import numpy as np

from purediscover_quantum import QuantumOracle
from purediscover_dual import search_dual


def test_quantum_oracle() -> int:
    print("=== (c) quantum oracle ===")
    oracle = QuantumOracle(
        hamiltonian_spec=[("Bx", "X"), ("Bz", "Z")],
        n_qubits=1, initial_state="0", observable="Z", time_param="t")

    def analytic(Bx, Bz, t):
        B = math.sqrt(Bx * Bx + Bz * Bz)
        if B < 1e-12:
            return 1.0
        return 1.0 - 2.0 * (Bx ** 2 / (B * B)) * (math.sin(B * t) ** 2)

    max_err = 0.0
    for Bx, Bz, t in [(0.5, 0.3, 1.0), (0.9, 0.1, 2.0), (0.2, 0.8, 0.7)]:
        yn = oracle.measure({"Bx": Bx, "Bz": Bz, "t": t})
        ya = analytic(Bx, Bz, t)
        e = abs(yn - ya)
        max_err = max(max_err, e)
        print(f"  Bx={Bx} Bz={Bz} t={t}  numeric={yn:+.6f}  "
              f"analytic={ya:+.6f}  err={e:.2e}")
    ok = max_err < 1e-10
    print(f"  max error: {max_err:.2e}  ({'PASS' if ok else 'FAIL'})")

    out = oracle.to_csv(
        str(_ROOT / "datasets/quantum/qubit_rabi.csv"),
        param_grid={
            "Bx": list(np.linspace(0.1, 1.0, 5)),
            "Bz": list(np.linspace(0.0, 1.0, 5)),
            "t":  list(np.linspace(0.0, 3.0, 10)),
        })
    n_rows = sum(1 for _ in open(out)) - 1
    print(f"  wrote {out}  ({n_rows} rows)")
    return 0 if ok else 1


def test_dual_search() -> int:
    print("\n=== (b) dual representation search ===")
    rng = np.random.default_rng(0)
    n = 500
    a = rng.uniform(0.5, 3.0, n)
    b = rng.uniform(0.5, 3.0, n)
    y = a * b
    data = [({"a": float(a[i]), "b": float(b[i])}, float(y[i])) for i in range(n)]

    result = search_dual(data, ["a", "b"], min_improvement=1.10,
                        time_budget_direct=5.0, time_budget_compose=10.0)
    print(f"  baseline_law : {result['baseline_law']}")
    print(f"  baseline_mdl : {result['baseline_mdl']:.3f}")
    print(f"  attempted    : {result['attempted']}  valid: {result['valid']}")
    if result["dual"]:
        d = result["dual"]
        print(f"  DUAL FOUND   : transforms={d.transforms}")
        print(f"                 law={d.law_str}")
        print(f"                 mdl={d.mdl:.3f}  improvement={d.improvement:.2f}x")
    else:
        print("  no dual with lower MDL — baseline likely minimal")
    return 0


def main() -> int:
    rc1 = test_quantum_oracle()
    rc2 = test_dual_search()
    return rc1 or rc2


if __name__ == "__main__":
    raise SystemExit(main())
