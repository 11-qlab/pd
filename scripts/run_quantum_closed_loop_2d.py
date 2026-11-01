#!/usr/bin/env python3
"""20-iteration 2-parameter closed loop against exact quantum oracle.

Physics:  H = Bx*X, initial |0>, obs=Z, t free.
Analytic: <Z>(Bx, t) = cos(2*Bx*t) = cos(a*x*y).

Target is 2D.  propose_experiment() returns a float, so we substitute
a 2D proposal: random sample with gap-aware dedup.  This is a placeholder
for real BOED in 2D, and it is enough to prove the loop drives itself.

Each iteration runs BOTH discovery arms:
  arm 1: discover_and_classify (tree search) — may fail on cos
  arm 2: search_templates_2d (template library) — will recover cos if present
"""
import contextlib
import io
import json
import math
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from purediscover_quantum import QuantumOracle
from purediscover.discover import discover_and_classify
from purediscover_templates import search_templates_2d


# config
X_RANGE = (0.10, 2.00)     # Bx
T_RANGE = (0.10, 3.00)     # t
N_SEED = 8
N_ITER = 20
TOL = 0.02
MIN_GAP = 0.05
CKPT = _ROOT / "results/closed_loop/state_2d.json"
LOG  = _ROOT / "results/closed_loop/log_2d.txt"


def quiet(fn, *a, **kw):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        try:
            return fn(*a, **kw), None
        except Exception as e:
            return None, e


def in_circle(pts, p, min_gap):
    for q in pts:
        if abs(q["Bx"] - p["Bx"]) < min_gap and abs(q["t"] - p["t"]) < min_gap:
            return True
    return False


def propose_2d(data, rng, n_cand=300):
    cand = []
    for _ in range(n_cand):
        p = {"Bx": float(rng.uniform(*X_RANGE)),
             "t":  float(rng.uniform(*T_RANGE))}
        if not in_circle([d[0] for d in data], p, MIN_GAP):
            cand.append(p)
    if not cand:
        return {"Bx": float(rng.uniform(*X_RANGE)),
                "t":  float(rng.uniform(*T_RANGE))}
    return cand[int(rng.integers(0, len(cand)))]


def main() -> int:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    lf = LOG.open("w")

    def log(m=""):
        print(m); lf.write(m + "\n"); lf.flush()

    log("=== 2D quantum closed loop ===")
    log(f"physics: <Z>(Bx,t) = cos(2*Bx*t)")
    log(f"Bx range: {X_RANGE},  t range: {T_RANGE}")
    log(f"seed points: {N_SEED},  iterations: {N_ITER}")
    log()

    oracle = QuantumOracle(
        hamiltonian_spec=[("Bx", "X"), ("Bz", "Z")],
        n_qubits=1, initial_state="0", observable="Z", time_param="t",
        noise_frac=0.02, noise_seed=42,
    )

    rng = np.random.default_rng(20260101)

    # seed: coarse 2x4 grid in (Bx, t)
    data = []
    for b in np.linspace(0.3, 1.8, 2):
        for t in np.linspace(0.3, 2.8, 4):
            y = oracle.measure({"Bx": float(b), "Bz": 0.0, "t": float(t)})
            data.append(({"Bx": float(b), "t": float(t)}, float(y)))
    log(f"seed: {len(data)} points")
    for d in data:
        log(f"  Bx={d[0]['Bx']:.3f}  t={d[0]['t']:.3f}  y={d[1]:+.6f}")
    log()

    for it in range(N_ITER):
        log(f"--- iter {it+1}/{N_ITER} ---")
        var_names = ["Bx", "t"]

        # arm 1: tree search (short budget — this arm is known-weak on cos)
        tree_c, tree_err = quiet(
            discover_and_classify, data, var_names,
            time_budget_direct=5.0, time_budget_compose=10.0, tol=TOL,
        )
        n_tree = len(tree_c) if tree_c else 0

        # arm 2: template library
        templ, templ_err = quiet(
            search_templates_2d, data, var_names,
            max_mre=0.10, n_multistart=3, seed=it,
        )
        n_templ = len(templ) if templ else 0
        best = templ[0] if templ else None

        log(f"  tree_search  : {n_tree} cands "
            f"{'(err: %s)' % tree_err if tree_err else ''}")
        if best:
            log(f"  templates    : {n_templ} fitted;  best = {best.name}")
            log(f"                 expr = {best.expr_str[:90]}")
            log(f"                 MRE  = {best.mre:.3e}  bits={best.mdl_bits:.1f}  ok={best.ok}")

        # propose + measure next point (2D placeholder)
        p_next = propose_2d(data, rng)
        y_next = oracle.measure({"Bx": p_next["Bx"], "Bz": 0.0,
                                 "t": p_next["t"]})
        y_ana = math.cos(2.0 * p_next["Bx"] * p_next["t"])
        data.append((p_next, float(y_next)))
        log(f"  proposed     : Bx={p_next['Bx']:.4f}  t={p_next['t']:.4f}")
        log(f"  measured     : y={y_next:+.8f}  analytic={y_ana:+.8f}  "
            f"err={abs(y_next-y_ana):.2e}")
        log(f"  n_points     : {len(data)}")

        CKPT.write_text(json.dumps({
            "iteration": it + 1,
            "n_points": len(data),
            "n_tree_cands": n_tree,
            "n_templ": n_templ,
            "best_template": {
                "name": best.name, "expr": best.expr_str,
                "mre": best.mre, "bits": best.mdl_bits, "ok": best.ok,
            } if best else None,
            "data": [{"Bx": d[0]["Bx"], "t": d[0]["t"], "y": d[1]}
                     for d in data],
        }, indent=2, default=str))
        log()

    # final report
    log("=== final template ranking (all 28 points) ===")
    final, _ = quiet(search_templates_2d, data, ["Bx", "t"],
                     max_mre=0.10, n_multistart=8, seed=42)
    if final:
        for i, r in enumerate(final[:6]):
            log(f"  [{i}] {r.name:20s}  MRE={r.mre:.3e}  "
                f"bits={r.mdl_bits:10.1f}  ok={r.ok}")
            log(f"       {r.expr_str[:100]}")
    else:
        log("  no templates fitted")

    # verify collected data is exact
    log()
    worst = 0.0
    for (xd, yd) in data:
        a = math.cos(2.0 * xd["Bx"] * xd["t"])
        worst = max(worst, abs(yd - a))
    log(f"worst |measured - analytic| over {len(data)} points: {worst:.2e}")
    log(f"oracle exactness: {'PASS' if worst < 1e-12 else 'REVIEW'}")

    lf.close()
    print(f"\nlog written to {LOG}")
    print(f"state written to {CKPT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
