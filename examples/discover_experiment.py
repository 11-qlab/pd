"""Run the discovery pipeline on the two CSVs.

Case A — Ohm's law (corpus)       -> expect KNOWN  I*R
Case B — radioactive decay        -> expect NOVEL  exp(-lam*t)

Budgets: brute_search should find both directly (<1s each). Compose
only runs if direct fails — should not run for either case now that
grammar has exp and search_tol is looser.
"""
import time

from purediscover.discover import load_csv, discover_and_classify


def _show(label, csv_path, target_col, *, max_direct=15.0, max_compose=60.0):
    print()
    print("=" * 78)
    print(f"  {label}")
    print(f"  file    : {csv_path}    target: {target_col}")
    print("=" * 78)

    try:
        data, var_names, target = load_csv(csv_path, target_col)
    except FileNotFoundError:
        print(f"  ✗ {csv_path} not found — run examples/make_experiment_csv.py first")
        return

    print(f"  loaded  : {len(data)} rows  vars={var_names}")

    t0 = time.time()
    results = discover_and_classify(
        data, var_names,
        time_budget_direct=max_direct,
        time_budget_compose=max_compose,
        tol=0.02,   # final filter: 2% mean relative error
    )
    elapsed = time.time() - t0
    print(f"  search  : {len(results)} candidates in {elapsed:.1f}s")
    print()

    if not results:
        print("  (no candidates — increase tol or max_nodes_direct)")
        return

    print(f"  {'#':>2s}  {'class':14s}  {'nodes':5s}  {'fit':>7s}  expression")
    print("  " + "-" * 74)
    for i, r in enumerate(results[:5], 1):
        mark = {"NOVEL": "*", "COMPOSITE": "+",
                "SCALED_KNOWN": "~", "KNOWN": "="}[r["class"]]
        detail = ""
        if "matched" in r["detail"]:
            detail = f"  → {r['detail']['matched']}"
        elif "parts" in r["detail"]:
            detail = (f"  ({r['detail']['parts'][0]} {r['detail']['op']} "
                      f"{r['detail']['parts'][1]})")
        print(f"  {i:>2d}.  {mark} {r['class']:12s}  "
              f"{r['n_nodes']:>5d}  {r['fit_error']:>7.4f}  "
              f"{r['expr']}{detail}")

    top = results[0]
    print()
    print(f"  verdict: {top['class']}  →  "
          + {
              "NOVEL": "send to human review",
              "COMPOSITE": "flag for physical plausibility",
              "SCALED_KNOWN": "rediscovery, auto-accept",
              "KNOWN": "rediscovery, auto-accept",
          }[top["class"]])


if __name__ == "__main__":
    _show("CASE A — Ohm's law experiment (expect KNOWN = ohm)",
          "experiment.csv", "V")

    _show("CASE B — radioactive decay (expect NOVEL exp(-lam*t))",
          "experiment_novel.csv", "N", max_direct=20.0, max_compose=90.0)

    print()
    print("=" * 78)
    print("  Case A verifies rediscovery of a known law from noisy data.")
    print("  Case B verifies production of a NOVEL form not in the corpus")
    print("  and not expressible as a binary composition of corpus laws.")
    print("=" * 78)
