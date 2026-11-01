#!/usr/bin/env python3
"""Dual search on the collected cos(Bx) data."""
import json, sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from purediscover_dual import search_dual

pts = json.loads((_ROOT / "results/closed_loop/state.json").read_text())["data"]
data = [({"Bx": p["Bx"]}, p["y"]) for p in pts]

result = search_dual(
    data, ["Bx"],
    min_improvement=1.10,
    time_budget_direct=15.0,
    time_budget_compose=30.0,
    tol=0.005,
    max_combo_size=1,
)
print(f"baseline: {result['baseline_law']}  mdl={result['baseline_mdl']:.2f}")
print(f"attempted: {result['attempted']}  valid: {result['valid']}")
if result["dual"]:
    d = result["dual"]
    print(f"DUAL: transform={d.transforms}")
    print(f"      law={d.law_str}")
    print(f"      mdl={d.mdl:.2f}  improvement={d.improvement:.2f}x")
else:
    print("no dual found")
