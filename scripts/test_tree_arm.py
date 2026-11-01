#!/usr/bin/env python3
"""Does tree_search_ext recover cos(2*Bx*t) from the 28 noisy points?"""
import contextlib, io, json, sys, time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from purediscover_ext import tree_search_ext


def main() -> int:
    pts = json.loads((_ROOT / "results/closed_loop/state_full.json").read_text())["data"]
    data = [({"Bx": p["Bx"], "t": p["t"]}, p["y"]) for p in pts]
    print(f"loaded {len(data)} noisy points")

    t0 = time.time()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        cands = tree_search_ext(
            data, ["Bx", "t"],
            ops=["add", "sub", "mul", "div", "pow", "sqrt", "exp", "log", "cos", "sin"],
            max_nodes=7, beam=400, tol=0.02, time_budget=60.0,
        )
    dt = time.time() - t0
    print(f"tree_search_ext took {dt:.1f}s, returned {len(cands)} candidates")
    print()
    for i, c in enumerate(cands[:10]):
        print(f"  [{i:02d}] bits={c['bits']:8.2f}  mre={c['mre']:.3e}  "
              f"n={c['n_nodes']}  {c['expr']}")
    print()
    truth = "cos(2"
    has_cos2 = any("cos" in c["expr"] and "2" in c["expr"] for c in cands[:20])
    has_cos_any = any("cos" in c["expr"] for c in cands)
    print(f"ground truth: <Z>(Bx,t) = cos(2*Bx*t)")
    print(f"  any cos at all          : {has_cos_any}")
    print(f"  cos with '2' (structural): {has_cos2}")
    if cands:
        print(f"  best MRE                : {min(c['mre'] for c in cands):.3e}")
    return 0 if has_cos_any else 1


if __name__ == "__main__":
    raise SystemExit(main())
