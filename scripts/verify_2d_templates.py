#!/usr/bin/env python3
"""Fit the 2D template library to the 28 collected points from the loop."""
import json, math, sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from purediscover_templates import search_templates_2d


def main() -> int:
    state = json.loads((_ROOT / "results/closed_loop/state_2d.json").read_text())
    pts = state["data"]
    data = [({"Bx": p["Bx"], "t": p["t"]}, p["y"]) for p in pts]
    print(f"loaded {len(data)} points from checkpoint")
    print(f"  Bx range: [{min(d[0]['Bx'] for d in data):.3f}, "
          f"{max(d[0]['Bx'] for d in data):.3f}]")
    print(f"  t  range: [{min(d[0]['t'] for d in data):.3f}, "
          f"{max(d[0]['t'] for d in data):.3f}]")
    print()

    results = search_templates_2d(
        data, ["Bx", "t"], max_mre=0.10, n_multistart=10, seed=42,
        verbose=True,
    )

    print(f"\n=== {len(results)} templates fitted ===")
    print(f"{'name':22s} {'mre':>12s} {'rms':>10s} {'bits':>10s} {'ok':>5s}")
    print("-" * 66)
    for r in results:
        bits = f"{r.mdl_bits:10.2f}" if math.isfinite(r.mdl_bits) else "       inf"
        print(f"{r.name:22s} {r.mre:12.3e} {r.rms:10.3e} {bits} {str(r.ok):>5s}")
        print(f"  expr: {r.expr_str[:100]}")

    if results:
        w = results[0]
        print(f"\n=== WINNER: {w.name} ===")
        print(f"  expr    : {w.expr_str}")
        print(f"  MRE     : {w.mre:.3e}")
        print(f"  params  : {w.params}")
        print(f"  MDL bits: {w.mdl_bits:.2f}")

    # ground truth check
    print(f"\n=== ground truth: <Z>(Bx,t) = cos(2*Bx*t) ===")
    if results:
        best = results[0]
        # reconstruct best at same points
        xs = [d[0]["Bx"] for d in data]
        ts = [d[0]["t"] for d in data]
        y_true = [math.cos(2.0 * x * t) for x, t in zip(xs, ts)]
        y_meas = [d[1] for d in data]
        worst = max(abs(a - b) for a, b in zip(y_true, y_meas))
        print(f"  worst |measured - cos(2*Bx*t)| = {worst:.2e}")
    return 0 if results and results[0].mre < 1e-6 else 1


if __name__ == "__main__":
    raise SystemExit(main())
