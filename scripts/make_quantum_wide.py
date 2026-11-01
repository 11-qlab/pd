#!/usr/bin/env python3
"""Generate wide-range cos(Bx) data where cos actually turns negative.

Bx in [0.05, 6.28] covers one full period of cos(Bx).
Includes the negative lobe (Bx > pi/2 ~ 1.57) — where cos is
distinguishable from any even positive surrogate like exp(-a*Bx^2).
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
from purediscover_quantum import QuantumOracle


def main() -> int:
    oracle = QuantumOracle(
        hamiltonian_spec=[("Bx", "X"), ("Bz", "Z")],
        n_qubits=1, initial_state="0", observable="Z", time_param="t",
    )

    # 60 points over [0.05, 6.28] — includes 3 sign changes of cos(Bx)
    grid = list(np.linspace(0.05, 6.28, 60))

    out = _ROOT / "datasets/quantum/qubit_cos_wide.csv"
    oracle.to_csv(str(out), param_grid={
        "Bx": grid, "Bz": [0.0], "t": [0.5],
    })

    # also overwrite the closed-loop checkpoint with this wider data
    import json
    data = []
    for bx in grid:
        y = oracle.measure({"Bx": bx, "Bz": 0.0, "t": 0.5})
        data.append(({"Bx": float(bx)}, float(y)))

    ckpt = _ROOT / "results/closed_loop/state.json"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    ckpt.write_text(json.dumps({
        "source": "wide-range cos(Bx)",
        "n_points": len(data),
        "data": [{"Bx": d[0]["Bx"], "y": d[1]} for d in data],
    }, indent=2))

    print(f"wrote {out}  ({len(grid)} points)")
    print(f"overwrote checkpoint with wide-range data")

    # sanity: show the sign changes
    ys = [d[1] for d in data]
    x_zeros = []
    for i in range(len(ys) - 1):
        if ys[i] * ys[i + 1] < 0:
            x_zeros.append((data[i][0]["Bx"] + data[i + 1][0]["Bx"]) / 2)
    print(f"sign changes near Bx = {[f'{x:.3f}' for x in x_zeros]}")
    print(f"  (expected near pi/2={np.pi/2:.3f}, 3*pi/2={3*np.pi/2:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
