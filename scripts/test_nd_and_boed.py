#!/usr/bin/env python3
"""Verify ND templates + BOED proposal on the 2D cos(2*Bx*t) case."""
import json, math, sys, time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
from purediscover_templates import search_templates_nd
from purediscover_ext import propose_boed_2d


def main() -> int:
    # load the state from the earlier loop run (37 noisy points)
    state = json.loads((_ROOT / "results/closed_loop/state_full.json").read_text())
    data = [({"Bx": p["Bx"], "t": p["t"]}, p["y"]) for p in state["data"]]
    print(f"loaded {len(data)} noisy points from state_full.json")
    print()

    # ---- TEST 1: ND templates ----------------------------------------
    print("=== TEST 1: ND templates (3D-equiv interface on 2D data) ===")
    t0 = time.time()
    results = search_templates_nd(data, ["Bx", "t"],
                                 max_order=3, max_mre=0.15,
                                 n_multistart=20, seed=42)
    dt = time.time() - t0
    print(f"  {len(results)} candidates fitted in {dt:.1f}s")
    for i, r in enumerate(results[:5]):
        print(f"  [{i}] {r.composite:12s} {r.name:22s}  "
              f"mae={r.mae:.3e}  bits={r.mdl_bits:8.1f}  ok={r.ok}")
        print(f"      {r.expr_str[:90]}")
    print()
    if results:
        w = results[0]
        truth = "cos(2"
        found_cos = "cos" in w.expr_str
        near_a = any(abs(abs(float(p)) - 1.0) < 0.1 for p in w.params if p)
        print(f"  WINNER: {w.expr_str[:80]}")
        print(f"  contains cos: {found_cos}    any param ≈1: {near_a}")
    print()

    # ---- TEST 2: BOED proposal ---------------------------------------
    print("=== TEST 2: BOED proposal from top-K candidates ===")
    # convert NDResult list to the format propose_boed_2d expects
    cands = []
    for r in results[:8]:
        # find the sympy expression (re-parse from string)
        try:
            import sympy as sp
            e = sp.sympify(r.expr_str)
            cands.append({"expr": r.expr_str, "sympy_expr": e,
                          "bits": r.mdl_bits})
        except Exception:
            continue
    print(f"  feeding {len(cands)} candidates to BOED proposer")

    for label, xr in [("narrow", (0.3, 1.8)), ("wide", (0.1, 2.0))]:
        p = propose_boed_2d(cands, data, ["Bx", "t"], xr, (0.3, 2.8),
                           n_grid=30, top_k=5, min_gap=0.05)
        print(f"  {label:7s} region: x={p['x']:.4f}  y={p['y']:.4f}  "
              f"disagreement={p.get('disagreement', 'NA')}")
    print()

    # ---- TEST 3: what BOED predicts at that point --------------------
    if cands:
        import sympy as sp
        p = propose_boed_2d(cands, data, ["Bx", "t"], (0.1, 2.0), (0.1, 3.0),
                           n_grid=30, top_k=5, min_gap=0.05)
        x_sym = sp.Symbol("Bx"); y_sym = sp.Symbol("t")
        print("=== TEST 3: predicted values at BOED point ===")
        print(f"  at Bx={p['x']:.4f}, t={p['y']:.4f}:")
        preds = []
        for c in cands[:5]:
            try:
                f = sp.lambdify([x_sym, y_sym], c["sympy_expr"], modules=["numpy"])
                v = float(f(p["x"], p["y"]))
                preds.append(v)
                print(f"    {v:+.4f}   {c['expr'][:60]}")
            except Exception:
                continue
        if len(preds) >= 2:
            print(f"  spread: {max(preds)-min(preds):.4f}")
            truth_v = math.cos(2*p["x"]*p["y"])
            print(f"  ground truth: {truth_v:+.4f}")
    print()
    print("=== SUMMARY ===")
    nd_ok = False
    if results:
        w = results[0]
        if "cos" in w.expr_str:
            nd_ok = True
        elif "sin" in w.expr_str:
            for param in (w.params or []):
                try:
                    if abs(abs(float(param)) - math.pi/2) < 0.15:
                        nd_ok = True
                        break
                except (TypeError, ValueError):
                    continue
    print(f"  ND templates recover cos form: {nd_ok}")
    print(f"  BOED returns a point outside training grid: "
          f"{not any(abs(p['x']-d[0]['Bx'])<0.05 and abs(p['y']-d[0]['t'])<0.05 for d in data)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
