#!/usr/bin/env python3
"""After adding trig features: does the sparse arm find cos(2*Bx*t)?"""
import json, math, sys, time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from purediscover_ext import tree_search_ext


def main() -> int:
    state = json.loads((_ROOT / "results/closed_loop/state_full.json").read_text())
    data = [({"Bx": p["Bx"], "t": p["t"]}, p["y"]) for p in state["data"]]
    print(f"loaded {len(data)} points from state_full.json")
    print(f"Bx range [{min(d[0]['Bx'] for d in data):.3f}, "
          f"{max(d[0]['Bx'] for d in data):.3f}]")
    print(f"t  range [{min(d[0]['t'] for d in data):.3f}, "
          f"{max(d[0]['t'] for d in data):.3f}]")
    print()

    t0 = time.time()
    cands = tree_search_ext(
        data, ["Bx", "t"],
        max_features=3, degree=2, include_trig=True,
        tol=0.02, time_budget=60.0,
    )
    dt = time.time() - t0
    print(f"sparse arm returned {len(cands)} candidates in {dt:.1f}s")
    print()
    for i, c in enumerate(cands[:10]):
        has_cos = "cos" in c["expr"]
        print(f"  [{i:02d}] mae={c['mre']:.3e}  bits={c['bits']:8.1f}  "
              f"n={c['n_nodes']}  {'COS' if has_cos else '   '}  "
              f"{c['expr'][:70]}")
    print()
    has_cos_any = any("cos" in c["expr"] for c in cands)
    best = cands[0] if cands else None
    print(f"=== verdict ===")
    print(f"  sparse arm finds cos expression: {has_cos_any}")
    print(f"  best mae                        : "
          f"{best['mre'] if best else 'NA'}")
    if best and has_cos_any:
        # check if any cos has frequency ~2
        for c in cands:
            if "cos" in c["expr"]:
                try:
                    # coefficients stored in expr text
                    import re
                    coefs = [float(x) for x in re.findall(r'(\d+\.\d+)', c["expr"])]
                    if any(abs(a - 2.0) < 0.15 for a in coefs):
                        print(f"  coeff with ~2.0 detected    : True")
                        break
                except Exception:
                    pass
    return 0 if has_cos_any else 1


if __name__ == "__main__":
    raise SystemExit(main())
