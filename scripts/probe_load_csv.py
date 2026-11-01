#!/usr/bin/env python3
"""Discover exactly what purediscover.discover.load_csv returns."""
import sys
from pathlib import Path

from purediscover.discover import load_csv

CSV = Path("datasets/susy_M_R.csv")
TGT = "M_R"


def describe(x, depth=0, max_depth=2):
    pad = "  " * depth
    t = type(x).__name__
    try:
        n = len(x)
    except TypeError:
        n = None
    print(f"{pad}type={t}  len={n}")
    if depth >= max_depth:
        return
    if isinstance(x, tuple):
        for i, item in enumerate(x):
            print(f"{pad}  [{i}]")
            describe(item, depth + 2, max_depth)
    elif isinstance(x, dict):
        for k in list(x.keys())[:10]:
            print(f"{pad}  [{k!r}]")
            describe(x[k], depth + 2, max_depth)
    elif hasattr(x, "__dataclass_fields__"):
        for f in x.__dataclass_fields__:
            print(f"{pad}  .{f}")
            describe(getattr(x, f), depth + 2, max_depth)
    elif hasattr(x, "__len__") and n and n <= 20:
        print(f"{pad}  values={x!r}")
    elif hasattr(x, "__len__") and n:
        print(f"{pad}  first_three={x[:3]!r}")


def main() -> int:
    if not CSV.exists():
        print(f"missing {CSV}", file=sys.stderr)
        return 1
    print(f"calling load_csv({CSV!s}, target_col={TGT!r})\n")
    try:
        result = load_csv(str(CSV), target_col=TGT)
    except Exception as e:
        print(f"FAIL during call: {type(e).__name__}: {e}", file=sys.stderr)
        return 1

    print("=== return value shape ===")
    describe(result)
    print()

    print("=== top-level repr (truncated) ===")
    r = repr(result)
    print(r[:800] + ("..." if len(r) > 800 else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
