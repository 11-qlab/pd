"""Real-data discovery test: Kepler's Third Law.

Data: actual semi-major axes and orbital periods of the 8 planets
(source: NASA JPL planetary fact sheets). The law P^2 = a^3 is
universally known to humans but is NOT in purediscover's 56-law corpus.

Expected: NOVEL (or COMPOSITE if the classifier finds an algebraic
coincidence), expression close to a**(3/2).
"""
import csv
from pathlib import Path

from purediscover.discover import load_csv, discover_and_classify

# --- real planetary data (JPL) -------------------------------------------
PLANETS = [
    # name, semi-major axis (AU), orbital period (yr)
    ("Mercury", 0.387098, 0.240846),
    ("Venus",   0.723332, 0.615198),
    ("Earth",   1.000000, 1.000017),
    ("Mars",    1.523679, 1.880847),
    ("Jupiter", 5.204267, 11.8618),
    ("Saturn",  9.582017, 29.4571),
    ("Uranus",  19.18916, 84.0205),
    ("Neptune", 30.06992, 164.770),
]

path = Path("datasets/kepler_third_law.csv")
path.parent.mkdir(exist_ok=True)
with path.open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["a", "P"])
    for _, a, P in PLANETS:
        w.writerow([f"{a:.6f}", f"{P:.6f}"])

print(f"wrote {path}  ({len(PLANETS)} planets)")
print("law: P = a^(3/2)  (Kepler's third law, not in corpus)")

# --- run discovery -------------------------------------------------------
data, var_names, target = load_csv(path, target_col="P")
print(f"loaded {len(data)} rows, vars={var_names}, target={target}\n")

results = discover_and_classify(
    data, var_names,
    max_nodes_direct=8,
    time_budget_direct=60,
    time_budget_compose=180,
)

if not results:
    print("(no candidates — try increasing max_nodes or time budgets)")
else:
    print(f"{'#':>2s}  {'class':14s}  {'nodes':5s}  {'fit':>8s}  expression")
    print("-" * 70)
    for i, r in enumerate(results[:5], 1):
        mark = {"NOVEL": "*", "COMPOSITE": "+",
                "SCALED_KNOWN": "~", "KNOWN": "="}[r["class"]]
        print(f"{i:>2d}.  {mark} {r['class']:12s}  "
              f"{r['n_nodes']:>5d}  {r['fit_error']:>8.5f}  {r['expr']}")
    top = results[0]
    print(f"\nverdict: {top['class']}  →  "
          + {"NOVEL": "send to human review",
             "COMPOSITE": "flag for plausibility",
             "SCALED_KNOWN": "rediscovery, auto-accept",
             "KNOWN": "rediscovery, auto-accept"}[top["class"]])
