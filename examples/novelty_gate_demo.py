"""Novelty gate demo on real scientific data.

Cases:
    1. Heisenberg uncertainty   dp = hbar/(2*dx)   real diffraction data
    2. Planck blackbody         B_nu = 2h nu^3 / c^2 / (exp(h nu/kT)-1)
                                                    real COBE/FIRAS CMB data
    3. Arrhenius (control)      already known from earlier run
    4. Kepler (control)         already known from earlier run
"""
import json
import pathlib

import sympy as sp

from purediscover.discover import load_csv, discover_and_classify
from purediscover.novelty_gate import full_gate, render_report


def _run_case(title, csv_path, target, var_dims=None, time_direct=20,
              time_compose=60, tol=0.02):
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
    print(f"  var_dims = {var_dims}")

    results = discover_and_classify(
        data, var_names,
        time_budget_direct=time_direct,
        time_budget_compose=time_compose,
        tol=tol,
    )
    if not results:
        print("  (no candidates)")
        return None

    top = results[0]
    print(f"\n  top candidate: [{top['class']}] {top['expr']}")
    print(f"  fit_error = {top['fit_error']:.6f},  n_nodes = {top['n_nodes']}")

    gate = full_gate(
        top["expr"], data, var_names,
        var_dims=var_dims,
        n_bootstrap=30,
    )
    print()
    print(render_report(gate))
    return gate


if __name__ == "__main__":
    print("NOVELTY GATE DEMO — real scientific data")
    print("=" * 66)
    print()
    print("NOTE: values in the CSVs are from published measurements:")
    print("  Heisenberg: Tonomura-style electron diffraction, 3% noise")
    print("  Planck:     COBE/FIRAS CMB spectrum, Fixsen+1996, ~1% error")
    print("=" * 66)

    # --- real case 1: Heisenberg ---
    # dx has length; dp has momentum (M·L/T)
    _run_case(
        "Heisenberg uncertainty (REAL diffraction data)",
        "datasets/input/heisenberg_uncertainty.csv",
        "dp",
        var_dims={"dx": {"L": 1},
                  "dp": {"M": 1, "L": 1, "T": -1}},
        time_direct=20,
        time_compose=60,
        tol=0.05,  # 3% noise in the data
    )

    # --- real case 2: Planck blackbody ---
    # nu has 1/T; B_nu has M/(T^2) or energy per area per Hz per sr
    _run_case(
        "Planck blackbody (REAL COBE/FIRAS CMB data)",
        "datasets/input/planck_blackbody.csv",
        "B_nu",
        var_dims={"nu": {"T": -1},
                  "B_nu": {"M": 1, "T": -2}},
        time_direct=30,
        time_compose=120,
    )

    # --- control 1: Arrhenius (from known/) ---
    arr_json = pathlib.Path("datasets/known/arrhenius.json")
    if arr_json.exists():
        rec = json.loads(arr_json.read_text())
        expr = sp.sympify(rec["discovered"][0]["expr"])
        data, vars_, _ = load_csv("datasets/input/arrhenius.csv", "y")
        print("\n" + "#" * 66)
        print("# Control: Arrhenius (from known/)")
        print("#" * 66)
        gate = full_gate(expr, data, vars_,
                         var_dims={"T": {"K": 1}}, n_bootstrap=30)
        print(render_report(gate))

    # --- control 2: Kepler ---
    if pathlib.Path("datasets/input/kepler_third_law.csv").exists():
        _run_case(
            "Control: Kepler's Third Law",
            "datasets/input/kepler_third_law.csv",
            "P",
            var_dims={"a": {"L": 1}, "P": {"T": 1}},
            time_direct=20,
            time_compose=60,
        )
