"""Verdict contrast demo: LOW / MEDIUM / REJECT.

Cases:
    1. Arrhenius (existing)     — known law  → LOW
    2. Kepler (existing)        — known law  → LOW
    3. Novel: A/x + B           — not in corpus → MEDIUM
    4. Noise                    — no stable fit → REJECT or "no candidates"

The point: the gate must distinguish the three outcomes correctly.
A form that fits well but isn't a known law → MEDIUM.
A form that fits well but matches a known law → LOW.
A form that doesn't fit stably → REJECT (or pipeline finds nothing).
"""
import csv
import pathlib
import math
import random

import sympy as sp

from purediscover.discover import load_csv, discover_and_classify
from purediscover.novelty_gate import full_gate, render_report


def _make_novel_csv():
    """y = 2/x + 1 — A/x + B, offset inverse.

    Widened x range (0.2–20) so a pure power law surrogate cannot
    capture both asymptotes. The offset form 2/x + 1 is uniquely
    correct at this range; a surrogate A/x^0.3 fits badly at
    intermediate x values.
    """
    path = pathlib.Path("datasets/input/novel_ax_b.csv")
    path.parent.mkdir(exist_ok=True)
    rng = random.Random(3)
    log_lo = math.log(0.2)
    log_hi = math.log(20.0)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["x", "y"])
        for _ in range(30):
            # Log-uniform sampling: gives ~10 samples per decade, so
            # the Laurent fit sees enough low-x points to constrain the
            # 1/x coefficient. Uniform [0.2, 20] puts ~1 sample below
            # x=1, leaving the fit underdetermined.
            x = math.exp(rng.uniform(log_lo, log_hi))
            # Multiplicative noise — matches real measurement precision
            y = (2.0 / x + 1.0) * (1.0 + rng.gauss(0.0, 0.02))
            w.writerow([f"{x:.6f}", f"{y:.6f}"])
    return path


def _make_noise_csv():
    """30 points of uniform noise in y = rand(0,1)."""
    path = pathlib.Path("datasets/input/pure_noise.csv")
    rng = random.Random(11)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["x", "y"])
        for _ in range(30):
            x = rng.uniform(0.1, 10.0)
            y = rng.uniform(0.0, 1.0)
            w.writerow([f"{x:.6f}", f"{y:.6f}"])
    return path


def _run(title, csv_path, target, *, var_dims=None, tol=0.02,
         time_direct=20, time_compose=60):
    print("\n" + "#" * 66)
    print(f"# {title}")
    print(f"# file={csv_path}  target={target}  tol={tol}")
    print("#" * 66)

    try:
        data, var_names, _ = load_csv(csv_path, target_col=target)
    except FileNotFoundError:
        print(f"  ✗ {csv_path} not found")
        return None

    print(f"  loaded {len(data)} rows, vars={var_names}")

    results = discover_and_classify(
        data, var_names,
        time_budget_direct=time_direct,
        time_budget_compose=time_compose,
        tol=tol,
    )
    if not results:
        print("  (no candidates — pipeline correctly gave up)")
        return "no_candidates"

    top = results[0]
    print(f"\n  top candidate: [{top['class']}] {top['expr']}")
    print(f"  fit_error = {top['fit_error']:.6f}, n_nodes = {top['n_nodes']}")

    gate = full_gate(top["expr"], data, var_names,
                     var_dims=var_dims, n_bootstrap=30)
    print()
    print(render_report(gate))
    return gate["verdict"]


def _summary(rows):
    print("\n" + "=" * 66)
    print("VERDICT SUMMARY")
    print("=" * 66)
    print(f"  {'case':30s}  {'verdict':14s}")
    print("  " + "-" * 60)
    for name, v in rows:
        marker = {"LOW": "★  known", "MEDIUM": "★★ novel",
                  "HIGH": "★★★ novel+replicated", "REJECT": "✗  reject",
                  "no_candidates": "✗  gave up"}
        print(f"  {name:30s}  {marker.get(v, v)}")


if __name__ == "__main__":
    print("VERDICT CONTRAST DEMO")
    print("=" * 66)
    print("Shows LOW (known), MEDIUM (novel), and REJECT (noise) on")
    print("real and synthetic data.")
    print("=" * 66)

    rows = []

    # Known law: Arrhenius
    arr = pathlib.Path("datasets/input/arrhenius.csv")
    if arr.exists():
        v = _run("Arrhenius (known)", arr, "y",
                 var_dims={"T": {"K": 1}}, tol=0.05,
                 time_direct=20, time_compose=60)
        rows.append(("Arrhenius (known)", v or "?"))

    # Known law: Kepler
    kep = pathlib.Path("datasets/input/kepler_third_law.csv")
    if kep.exists():
        v = _run("Kepler (known)", kep, "P",
                 var_dims={"a": {"L": 1}, "P": {"T": 1}},
                 tol=0.02, time_direct=20, time_compose=60)
        rows.append(("Kepler (known)", v or "?"))

    # Novel: A/x + B
    novel_csv = _make_novel_csv()
    v = _run("Novel: A/x + B", novel_csv, "y",
             tol=0.03, time_direct=20, time_compose=60)
    rows.append(("Novel: A/x + B", v or "?"))

    # Noise
    noise_csv = _make_noise_csv()
    v = _run("Pure noise", noise_csv, "y",
             tol=0.02, time_direct=20, time_compose=60)
    rows.append(("Pure noise", v or "?"))

    _summary(rows)
