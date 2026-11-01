#!/usr/bin/env python3
"""Run purediscover discovery on the cleaned SUSY CSV.

load_csv returns (data, var_names, target_col_name) — 3-tuple.
discover_and_classify takes (data, var_names, *, tol=...).
"""
import json
import sys
from pathlib import Path

from purediscover.discover import load_csv, discover_and_classify

CSV = Path("datasets/susy_M_R.csv")
TGT = "M_R"
OUT = Path("results/susy_discovery.json")
TOL = 0.10   # 2% default is for synthetic benchmark; SUSY needs more slack


def _get(d, *keys, default="?"):
    if isinstance(d, dict):
        for k in keys:
            if k in d:
                return d[k]
    return default


def main() -> int:
    if not CSV.exists():
        print(f"missing {CSV}", file=sys.stderr)
        return 1

    print(f"[1/3] loading {CSV}  target={TGT!r}")
    data, var_names, target_name = load_csv(str(CSV), target_col=TGT)
    print(f"      n_points={len(data)}  var_names={var_names}  target={target_name!r}")

    print(f"[2/3] discover_and_classify (tol={TOL}) ...")
    candidates = discover_and_classify(data, var_names, tol=TOL)
    print(f"      returned {len(candidates)} candidates")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(candidates, indent=2, default=str))
    print(f"[3/3] wrote {OUT}")
    print()

    for i, c in enumerate(candidates[:15]):
        expr = _get(c, "expression", "expr", "formula")
        cls  = _get(c, "classification", "novelty", "class")
        err  = _get(c, "error", "score", "rmse")
        print(f"  [{i:02d}] {str(cls):10s} err={err}")
        print(f"        {expr}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
