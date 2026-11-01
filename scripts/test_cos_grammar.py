#!/usr/bin/env python3
"""Re-run discovery on the 15 collected quantum points, with cos/sin added."""
import contextlib
import io
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from purediscover.discover import discover_and_classify

CKPT = _ROOT / "results/closed_loop/state.json"


def quiet(fn, *a, **kw):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        return fn(*a, **kw)


def main() -> int:
    state = json.loads(CKPT.read_text())
    pts = state["data"]
    data = [({"Bx": p["Bx"]}, p["y"]) for p in pts]
    print(f"loaded {len(data)} points from checkpoint")

    ops_sets = {
        "baseline (no trig)": ("add", "sub", "mul", "div", "pow", "sqrt", "log", "exp"),
        "with cos+sin":        ("add", "sub", "mul", "div", "pow", "sqrt", "log", "exp", "cos", "sin"),
    }

    for label, ops in ops_sets.items():
        print(f"\n=== {label} ===")
        try:
            cands = quiet(
                discover_and_classify,
                data, ["Bx"],
                grammar_ops=ops,
                time_budget_direct=30.0,
                time_budget_compose=60.0,
                tol=0.01,   # tight tolerance: force exact recovery
            )
        except Exception as e:
            print(f"  failed: {type(e).__name__}: {e}")
            continue
        if not cands:
            print("  no candidates")
            continue
        print(f"  {len(cands)} candidates:")
        for i, c in enumerate(cands[:8]):
            print(f"    [{i}] {c.get('expr', '?')}")
            print(f"        fit_error={c.get('fit_error'):.6e}  "
                  f"n_nodes={c.get('n_nodes')}  class={c.get('class')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
