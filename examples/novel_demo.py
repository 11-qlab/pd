"""End-to-end novel discovery demonstration.

Three cases with the same engine, different data:
    1. kinetic_energy (in corpus)              -> KNOWN
    2. kinetic_energy + momentum               -> COMPOSITE
    3. kinetic_energy / (1 + r)                -> NOVEL

The point: the *engine* cannot tell these apart — all three are
"expressions that fit data." The *novelty classifier* separates them,
using the corpus as the reference frame. Only case 3 is a candidate
for genuine novel discovery.
"""
import random
import time

import sympy as sp

from purediscover.symbolic import Grammar
from purediscover.rich_search import brute_search
from purediscover.novelty import classify_novelty
from purediscover.resource.laws import REGISTRY


# --- data generation -----------------------------------------------------

def _generate(target, var_names, n=40, seed=42, noise=0.0):
    rng = random.Random(seed)
    data = []
    tries = 0
    while len(data) < n and tries < n * 20:
        tries += 1
        env = {v: rng.uniform(0.5, 3.0) for v in var_names}
        try:
            y = float(target.subs({sp.Symbol(k): v for k, v in env.items()}))
        except (ZeroDivisionError, TypeError, ValueError):
            continue
        if not (1e-6 < abs(y) < 1e6):
            continue
        if noise > 0:
            y += rng.gauss(0, noise * abs(y))
        data.append((env, y))
    return data


# --- discovery (same engine for all cases) -------------------------------

def _grammar(var_names):
    return Grammar(
        variables=var_names,
        constants=[1.0],
        max_depth=4,
        ops=["add", "sub", "mul", "div", "pow", "sqrt", "log", "exp"],
    )


def _discover(data, var_names, max_nodes=12, time_budget=20.0):
    return brute_search(
        _grammar(var_names), data,
        max_nodes=max_nodes,
        max_per_level=50000,
        tol=0.02,
        time_budget=time_budget,
    )


# --- display -------------------------------------------------------------

_MARKS = {"KNOWN": "=", "SCALED_KNOWN": "~", "COMPOSITE": "+", "NOVEL": "*"}


def _present(label, target, var_names, max_nodes=12, time_budget=20.0):
    print()
    print("=" * 78)
    print(f"  {label}")
    print(f"  target   : {target}")
    print(f"  variables: {var_names}")
    print("=" * 78)

    data = _generate(target, var_names, n=40, seed=42)
    print(f"  data     : {len(data)} points")

    t0 = time.time()
    cands = _discover(data, var_names, max_nodes=max_nodes,
                      time_budget=time_budget)
    elapsed = time.time() - t0
    print(f"  search   : {len(cands)} candidates in {elapsed:.1f}s")

    if not cands:
        print("  (increase max_nodes or time_budget)")
        return

    # classify top 5 unique expressions
    seen = set()
    shown = 0
    for c in cands:
        expr = c["expr"]
        key = str(expr)
        if key in seen:
            continue
        seen.add(key)
        n_nodes = sum(1 for _ in sp.preorder_traversal(expr))
        novelty = classify_novelty(expr, REGISTRY)
        mark = _MARKS[novelty["class"]]

        detail = ""
        if "matched" in novelty:
            detail = f"= {novelty['matched']}"
        elif "parts" in novelty:
            detail = (f"{novelty['parts'][0]} {novelty['op']} "
                      f"{novelty['parts'][1]}")

        print(f"\n    [{shown+1}] {mark} {novelty['class']:12s}  {expr}")
        print(f"         nodes={n_nodes}  {detail}")
        shown += 1
        if shown >= 5:
            break


# --- main ----------------------------------------------------------------

if __name__ == "__main__":
    m, v, r = sp.symbols("m v r")

    print("Novel discovery demo")
    print("Three data sources, one engine, three novelty classes.")
    print()
    print("Ruling:")
    print("  =  KNOWN         already a corpus law       (filter OUT)")
    print("  ~  SCALED_KNOWN  constant × a corpus law    (filter OUT)")
    print("  +  COMPOSITE     binary mix of two laws     (flag for review)")
    print("  *  NOVEL         not in or composed of      (investigate)")

    # Case 1: KNOWN
    _present(
        "CASE 1 — rediscovery of a corpus law",
        m * v ** 2,
        ["m", "v"],
    )

    # Case 2: COMPOSITE
    _present(
        "CASE 2 — a trivial composition of two corpus laws",
        m * v ** 2 + m * v,
        ["m", "v"],
        max_nodes=10,
    )

    # Case 3: NOVEL
    _present(
        "CASE 3 — a structurally novel form",
        m * v ** 2 / (1 + r),
        ["m", "v", "r"],
        max_nodes=14,
        time_budget=30.0,
    )

    print()
    print("=" * 78)
    print("  The engine found all three cases. The classifier separates them.")
    print("  In the flywheel, only NOVEL candidates reach human review;")
    print("  COMPOSITE candidates go to a separate queue for physical plausibility;")
    print("  KNOWN and SCALED_KNOWN are auto-accepted as rediscoveries.")
    print("=" * 78)
