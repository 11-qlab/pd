#!/usr/bin/env python3
"""Full 2D loop: templates + sparse + convergence, using MAE as primary metric."""
import contextlib
import io
import json
import math
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from purediscover_quantum import QuantumOracle
from purediscover_ext import ConvergenceMonitor, tree_search_ext
from purediscover_templates import search_templates_2d


X_RANGE = (0.10, 2.00)
T_RANGE = (0.10, 3.00)
N_SEED = 8
MAX_ITER = 30
TOL = 0.02
MIN_GAP = 0.05
NOISE = 0.02
SEED = 20260101
CKPT = _ROOT / "results/closed_loop/state_full.json"
LOG  = _ROOT / "results/closed_loop/log_full.txt"


def quiet(fn, *a, **kw):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        try:
            return fn(*a, **kw), None
        except Exception as e:
            return None, e


def in_circle(pts, p, gap):
    return any(abs(q["Bx"]-p["Bx"]) < gap and abs(q["t"]-p["t"]) < gap for q in pts)


def propose_2d(data, rng, n_cand=300):
    pts = [d[0] for d in data]
    cand = []
    for _ in range(n_cand):
        p = {"Bx": float(rng.uniform(*X_RANGE)),
             "t":  float(rng.uniform(*T_RANGE))}
        if not in_circle(pts, p, MIN_GAP):
            cand.append(p)
    if not cand:
        return {"Bx": float(rng.uniform(*X_RANGE)),
                "t":  float(rng.uniform(*T_RANGE))}
    return cand[int(rng.integers(0, len(cand)))]


def pick_best(tree_cands, tmpl_cands):
    pool = []
    for c in (tree_cands or []):
        pool.append({
            "source": "tree",
            "expr": c.get("expr", ""),
            "bits": float(c.get("bits", float("inf"))),
            "mae":  float(c.get("mae", c.get("mre", c.get("fit_error", float("inf"))))),
        })
    for t in (tmpl_cands or []):
        pool.append({
            "source": "templ",
            "expr": t.expr_str,
            "bits": float(t.mdl_bits),
            "mae":  float(t.mae),
        })
    pool = [p for p in pool if math.isfinite(p["bits"])]
    if not pool:
        return None
    pool.sort(key=lambda p: (p["bits"], p["mae"]))
    return pool[0]


def main() -> int:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    lf = LOG.open("w")

    def log(m=""):
        print(m)
        lf.write(m + "\n")
        lf.flush()

    log("=== FULL 2D loop: templates + sparse + convergence (MAE metric) ===")
    log(f"physics  : <Z>(Bx,t) = cos(2*Bx*t)")
    log(f"noise    : {NOISE*100:.1f}%   max_iter : {MAX_ITER}")
    log()

    oracle = QuantumOracle(
        hamiltonian_spec=[("Bx", "X"), ("Bz", "Z")],
        n_qubits=1, initial_state="0", observable="Z", time_param="t",
        noise_frac=NOISE, noise_seed=SEED,
    )
    rng = np.random.default_rng(SEED)

    data = []
    for b in np.linspace(0.3, 1.8, 2):
        for t in np.linspace(0.3, 2.8, 4):
            y = oracle.measure({"Bx": float(b), "Bz": 0.0, "t": float(t)})
            data.append(({"Bx": float(b), "t": float(t)}, float(y)))
    log(f"seed: {len(data)} points\n")

    monitor = ConvergenceMonitor(patience=5, mae_epsilon=5e-3, bits_epsilon=2.0)

    stopped_at = None
    for it in range(MAX_ITER):
        log(f"--- iter {it+1}/{MAX_ITER} ---")

        tree_c, tree_err = quiet(
            tree_search_ext, data, ["Bx", "t"],
            max_features=4, degree=2,
            tol=TOL, time_budget=25.0,
        )
        log(f"  tree_search  : {len(tree_c) if tree_c else 0} cands"
            + (f"  ERR: {tree_err}" if tree_err else ""))

        tmpl_c, tmpl_err = quiet(
            search_templates_2d, data, ["Bx", "t"],
            max_mre=0.15, n_multistart=6, seed=it,
        )
        if tmpl_c and tmpl_c[0].ok:
            t = tmpl_c[0]
            log(f"  templates    : top mae={t.mae:.3e}  mre={t.mre:.3e}  "
                f"bits={t.mdl_bits:.1f}")
            log(f"                 {t.expr_str[:80]}")
        else:
            log(f"  templates    : no winner yet")

        best = pick_best(tree_c, tmpl_c)
        if best:
            log(f"  BEST         : [{best['source']}] bits={best['bits']:.1f}  "
                f"mae={best['mae']:.3e}")
            log(f"                 {best['expr'][:100]}")
        else:
            log(f"  BEST         : (none)")

        status = monitor.update({
            "expr": best["expr"] if best else "",
            "bits": best["bits"] if best else float("inf"),
            "mae":  best["mae"]  if best else float("inf"),
        })
        log(f"  convergence  : {status['reason']}  streak={status['streak']}")

        CKPT.write_text(json.dumps({
            "iteration": it + 1,
            "n_points": len(data),
            "best_source": best["source"] if best else None,
            "best_expr": best["expr"] if best else None,
            "best_mae": best["mae"] if best else None,
            "best_bits": best["bits"] if best else None,
            "convergence": status,
            "data": [{"Bx": d[0]["Bx"], "t": d[0]["t"], "y": d[1]} for d in data],
        }, indent=2, default=str))

        if status["converged"]:
            stopped_at = it + 1
            log(f"  >>> CONVERGED at iteration {stopped_at}")
            break

        p_next = propose_2d(data, rng)
        y_next = oracle.measure({"Bx": p_next["Bx"], "Bz": 0.0, "t": p_next["t"]})
        data.append((p_next, float(y_next)))
        log(f"  measured     : Bx={p_next['Bx']:.4f}  t={p_next['t']:.4f}  "
            f"y={y_next:+.6f}")
        log(f"  n_points     : {len(data)}\n")

    if stopped_at is None:
        stopped_at = MAX_ITER
        log(f">>> iter cap reached at {MAX_ITER} without convergence\n")

    log("=== verification vs analytic  <Z> = cos(2*Bx*t) ===")
    ana = np.array([math.cos(2 * d[0]["Bx"] * d[0]["t"]) for d in data])
    meas = np.array([d[1] for d in data])
    mae_meas = float(np.mean(np.abs(meas - ana)))
    log(f"  MAE(measured vs analytic)  = {mae_meas:.4f}   "
        f"(expected for {NOISE*100:.0f}% noise: ~{NOISE*0.8:.3f})")
    log(f"  max|measured - analytic|   = {float(np.max(np.abs(meas - ana))):.4f}")
    log()

    log("=== convergence summary ===")
    s = monitor.summary()
    log(f"  iterations run : {s['n_iterations']}")
    log(f"  best expr      : {s.get('best_expr')}")
    log(f"  best mae       : {s.get('best_mae')}")
    log(f"  best bits      : {s.get('best_bits')}")
    log(f"  final expr     : {s.get('final_expr')}")
    log()
    log(f"  STOPPED AT ITERATION {stopped_at}  (cap {MAX_ITER})")

    lf.close()
    print(f"\nlog -> {LOG}\nstate -> {CKPT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
