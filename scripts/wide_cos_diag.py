#!/usr/bin/env python3
"""Three-level diagnostic on wide-range cos(Bx) data.

Test 1: discover_and_classify, default ops       -> baseline (likely bad fit)
Test 2: discover_and_classify, grammar_ops+cos   -> does the kwarg reach Grammar?
Test 3: brute_search with Grammar(ops+cos)       -> does the evaluator know cos?
"""
import contextlib
import inspect
import io
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from purediscover import Grammar
from purediscover.compose import brute_search
from purediscover.discover import discover_and_classify


def quiet(fn, *a, **kw):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        try:
            return fn(*a, **kw), None
        except Exception as e:
            return None, e


def show(cands, label):
    if cands is None:
        return
    print(f"  {len(cands)} candidates:")
    for i, c in enumerate(cands[:5]):
        e = c.get("expr", "?")
        fe = c.get("fit_error", "?")
        print(f"    [{i}] fit_error={fe}")
        print(f"        {e[:120]}")


def main() -> int:
    # --- load wide data --------------------------------------------------
    pts = json.loads((_ROOT / "results/closed_loop/state.json").read_text())["data"]
    data = [({"Bx": p["Bx"]}, p["y"]) for p in pts]
    xs = [p["Bx"] for p in pts]
    ys = [p["y"] for p in pts]
    n_neg = sum(1 for y in ys if y < 0)
    print(f"loaded {len(data)} points, Bx in [{min(xs):.2f}, {max(xs):.2f}]")
    print(f"y range [{min(ys):+.3f}, {max(ys):+.3f}],  points with y<0: {n_neg}")
    print()

    # --- test 1: default ops --------------------------------------------
    print("=== Test 1: discover_and_classify, default ops ===")
    c1, err = quiet(
        discover_and_classify, data, ["Bx"],
        time_budget_direct=30.0, time_budget_compose=60.0, tol=0.02,
    )
    if err is not None:
        print(f"  FAILED: {err}")
    else:
        show(c1, "default")
    print()

    # --- test 2: grammar_ops with cos -----------------------------------
    print("=== Test 2: discover_and_classify, grammar_ops + cos + sin ===")
    c2, err = quiet(
        discover_and_classify, data, ["Bx"],
        grammar_ops=("add", "sub", "mul", "div", "pow", "sqrt", "log", "exp",
                     "cos", "sin"),
        time_budget_direct=30.0, time_budget_compose=60.0, tol=0.02,
    )
    if err is not None:
        print(f"  FAILED: {err}")
    else:
        show(c2, "cos+sin")
    if c1 is not None and c2 is not None:
        same = all(str(a.get("expr")) == str(b.get("expr"))
                   for a, b in zip(c1, c2))
        print(f"  identical to Test 1: {same}")
        if same:
            print("  => grammar_ops kwarg is being dropped before Grammar is built")
    print()

    # --- test 3: brute_search with Grammar(ops+cos) ---------------------
    print("=== Test 3: brute_search directly, Grammar(ops + cos) ===")
    g = Grammar(variables=["Bx"],
                ops=["add", "sub", "mul", "div", "pow", "cos"],
                max_depth=4)
    print(f"  grammar.ops = {g.ops}")
    try:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            results = brute_search(g, data, max_nodes=14,
                                   time_budget=90.0, tol=0.02)
        print(f"  brute_search returned {len(results)} results")
        for i, r in enumerate(results[:5]):
            e = r.get("expr", r)
            print(f"    [{i}] {str(e)[:120]}")
            if "cos" in str(e):
                print(f"        ^ CONTAINS cos")
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")

    # --- inspect discover_and_classify source for grammar_ops handling --
    print()
    print("=== discover_and_classify: how is grammar_ops used? ===")
    try:
        src = inspect.getsource(discover_and_classify)
        for line in src.split("\n"):
            low = line.lower()
            if "grammar" in low or " ops" in line or "ops=" in line:
                print(f"  | {line.rstrip()}")
    except Exception as e:
        print(f"  (source unavailable: {e})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
