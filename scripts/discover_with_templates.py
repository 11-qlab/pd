#!/usr/bin/env python3
"""Template search on the 60-point wide-range cos(Bx) data."""
import json, sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from purediscover_templates import search_templates


def main() -> int:
    pts = json.loads((_ROOT / "results/closed_loop/state.json").read_text())["data"]
    data = [({"x": p["Bx"]}, p["y"]) for p in pts]
    print(f"loaded {len(data)} points, x in "
          f"[{min(d[0]['x'] for d in data):.3f}, "
          f"{max(d[0]['x'] for d in data):.3f}]")
    print()

    results = search_templates(data, ["x"], max_mre=0.05)

    print(f"=== {len(results)} templates fitted ===")
    print(f"{'name':22s} {'mre':>12s} {'rms':>10s} {'bits':>10s} {'ok':>3s}")
    print("-" * 62)
    for r in results:
        bits_s = f"{r.mdl_bits:10.2f}" if math.isfinite(r.mdl_bits) else "       inf"
        print(f"{r.name:22s} {r.mre:12.3e} {r.rms:10.3e} {bits_s} {str(r.ok):>3s}")
        print(f"  expr: {r.expr_str[:100]}")
    print()
    winner = results[0]
    print(f"=== WINNER: {winner.name} ===")
    print(f"  expr    : {winner.expr_str}")
    print(f"  MRE     : {winner.mre:.3e}")
    print(f"  params  : {winner.params}")
    print()
    print("Ground truth from quantum oracle: <Z>(Bx) = cos(Bx)")
    return 0 if winner.mre < 1e-6 else 1


if __name__ == "__main__":
    import math
    raise SystemExit(main())
