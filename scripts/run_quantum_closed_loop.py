#!/usr/bin/env python3
"""10-iteration closed loop against the quantum oracle.

Each iteration:
  1. discover_and_classify on the current data
  2. propose_experiment -> next x (via BOED/gap heuristic)
  3. QuantumOracle.measure at that x (exact quantum prediction)
  4. append to data, checkpoint

Physics: single qubit, H = Bx X (Bz=0), initial |0>, observable Z.
Analytic law: <Z>(Bx) = cos(2 Bx t). With t=0.5, <Z> = cos(Bx).
Grammar has no cos; polynomial surrogate is expected and accepted.
The point is that the loop runs unattended, appends real quantum data,
and improves its own fit — not that it recovers cos exactly.
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
from purediscover.closed_loop import discover_and_classify, propose_experiment


# --- config --------------------------------------------------------------
BZ = 0.0
T_EVOLVE = 0.5
SEED_BX = [0.1, 0.3, 0.5, 0.7, 0.9]
X_RANGE = (0.05, 1.0)
N_ITER = 10
TOL = 0.05
MAX_X_DUPLICATE = 1e-3
CKPT = _ROOT / "results/closed_loop/state.json"
LOG = _ROOT / "results/closed_loop/log.txt"
# -------------------------------------------------------------------------


def analytic(bx: float) -> float:
    """<Z>(t) for H = bx*X, initial |0>, t=T_EVOLVE. Bz=0 => cos(2 bx t)."""
    return math.cos(2.0 * bx * T_EVOLVE)


def quiet_call(fn, *args, **kwargs):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        try:
            return fn(*args, **kwargs), None
        except Exception as e:
            return None, e


def best_mdl(cands):
    """Cheap heuristic: shortest law string among candidates."""
    if not cands:
        return float("inf"), None
    best = min(cands, key=lambda c: len(str(c)))
    return len(str(best)), best


def coerce_x(x_raw, var_name, x_range):
    """Normalize whatever propose_experiment returns into a single float."""
    try:
        return float(np.clip(float(x_raw), *x_range))
    except (TypeError, ValueError):
        pass
    if isinstance(x_raw, dict):
        for k in (var_name, list(x_raw.keys())[0] if x_raw else None):
            if k is not None and k in x_raw:
                try:
                    return float(np.clip(float(x_raw[k]), *x_range))
                except (TypeError, ValueError):
                    continue
    if isinstance(x_raw, (list, tuple, np.ndarray)) and len(x_raw) > 0:
        try:
            return float(np.clip(float(x_raw[0]), *x_range))
        except (TypeError, ValueError):
            pass
    return float(np.random.uniform(*x_range))


def append_row(data, x, y, tol=MAX_X_DUPLICATE):
    for (xd, _yd) in data:
        if abs(xd["Bx"] - x) < tol:
            return False
    data.append(({"Bx": float(x)}, float(y)))
    return True


def main() -> int:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    log_fh = LOG.open("w")

    def log(msg=""):
        print(msg)
        log_fh.write(msg + "\n")
        log_fh.flush()

    log("=== quantum closed loop ===")
    log(f"physics     : H = Bx * X, Bz={BZ}, t={T_EVOLVE}, |0>, obs=Z")
    log(f"analytic    : <Z>(Bx) = cos(2*{T_EVOLVE}*Bx) = cos(Bx)")
    log(f"x_range     : {X_RANGE}")
    log(f"iterations  : {N_ITER}")
    log()

    oracle = QuantumOracle(
        hamiltonian_spec=[("Bx", "X"), ("Bz", "Z")],
        n_qubits=1, initial_state="0", observable="Z", time_param="t",
    )

    var_names = ["Bx"]
    data: list = []
    for bx in SEED_BX:
        y = oracle.measure({"Bx": bx, "Bz": BZ, "t": T_EVOLVE})
        data.append(({"Bx": float(bx)}, float(y)))
    log(f"seed: {len(data)} points")
    for (xd, yd) in data:
        log(f"  Bx={xd['Bx']:.4f}  y={yd:+.6f}  analytic={analytic(xd['Bx']):+.6f}")
    log()

    for it in range(N_ITER):
        log(f"--- iteration {it+1}/{N_ITER} ---")

        cands, err = quiet_call(
            discover_and_classify,
            data, var_names,
            time_budget_direct=5.0,
            time_budget_compose=10.0,
            tol=TOL,
        )
        if err is not None:
            log(f"  discover_and_classify FAILED: {err}")
            continue
        n_cands = len(cands) if cands else 0
        mdl_proxy, best_c = best_mdl(cands or [])
        log(f"  candidates   : {n_cands}")
        if best_c is not None:
            log(f"  best (short) : {best_c}")

        x_raw, err = quiet_call(
            propose_experiment,
            cands or [], data, var_names, X_RANGE,
            n_grid=200, top_k=3, random_state=it,
        )
        if err is not None:
            log(f"  propose_experiment FAILED: {err}  -> uniform fallback")
            x_raw = None
        x_next = coerce_x(x_raw, var_names[0], X_RANGE)

        y_next = oracle.measure({"Bx": x_next, "Bz": BZ, "t": T_EVOLVE})
        added = append_row(data, x_next, y_next)
        log(f"  proposed x   : {x_next:.6f}  "
            f"(analytic={analytic(x_next):+.6f})")
        log(f"  measured y   : {y_next:+.6f}  "
            f"(residual={y_next - analytic(x_next):+.2e})")
        log(f"  added        : {added}  total points: {len(data)}")

        state = {
            "iteration": it + 1,
            "n_points": len(data),
            "n_candidates": n_cands,
            "best_candidate": str(best_c) if best_c else None,
            "x_next": x_next,
            "y_next": y_next,
            "analytic": analytic(x_next),
            "data": [{"Bx": d[0]["Bx"], "y": d[1]} for d in data],
        }
        CKPT.write_text(json.dumps(state, indent=2, default=str))
        log(f"  checkpoint   : {CKPT}")
        log()

    # final: refit on the full 15-point dataset
    log("=== final fit ===")
    cands, err = quiet_call(
        discover_and_classify,
        data, var_names,
        time_budget_direct=30.0,
        time_budget_compose=60.0,
        tol=TOL,
    )
    if err is None:
        log(f"final candidates: {len(cands) if cands else 0}")
        if cands:
            for i, c in enumerate(cands[:10]):
                log(f"  [{i:02d}] {c}")

    # measured vs analytic at all collected points
    log()
    log("=== collected points: y (measured) vs analytic ===")
    worst = 0.0
    for (xd, yd) in sorted(data, key=lambda d: d[0]["Bx"]):
        a = analytic(xd["Bx"])
        e = abs(yd - a)
        worst = max(worst, e)
        log(f"  Bx={xd['Bx']:>8.5f}  y={yd:+.8f}  analytic={a:+.8f}  "
            f"err={e:.2e}")
    log()
    log(f"worst |y - analytic| over {len(data)} points: {worst:.2e}")
    log(f"target (oracle is exact): < 1e-12  ->  "
        f"{'PASS' if worst < 1e-12 else 'REVIEW'}")

    log_fh.close()
    print()
    print(f"full log written to {LOG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
