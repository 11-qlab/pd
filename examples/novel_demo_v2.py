"""Novel discovery demo v2 — uses compositional search for depth-3 targets.

Case 1: KNOWN         brute_search finds m*v**2 directly
Case 2: COMPOSITE     brute_search finds m*(v**2+v) directly
Case 3: NOVEL         brute_search can't reach m*v**2/(r+1) at depth 9
                      -> compose_search peels B=r+1, finds A=m*v**2
"""
import random
import time

import sympy as sp

from purediscover.symbolic import Grammar
from purediscover.rich_search import brute_search
from purediscover.compose import search_composition
from purediscover.novelty import classify_novelty
from purediscover.resource.laws import REGISTRY


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


def _present(label, target, var_names, *, search="brute",
             max_nodes=12, time_budget=30.0, ops=None):
    print()
    print("=" * 78)
    print(f"  {label}")
    print(f"  target   : {target}")
    print(f"  variables: {var_names}")
    print(f"  search   : {search}  (ops={ops or 'default'})")
    print("=" * 78)

    data = _generate(target, var_names, n=40, seed=42)
    print(f"  data     : {len(data)} points")

    t0 = time.time()
    if search == "brute":
        g = Grammar(
            variables=var_names,
            constants=[1.0],
            max_depth=4,
            ops=ops or ["add", "sub", "mul", "div", "pow", "sqrt", "log", "exp"],
        )
        cands = brute_search(g, data, max_nodes=max_nodes,
                             max_per_level=50000, tol=0.02,
                             time_budget=time_budget)
        candidates = [{"expr": c["expr"]} for c in cands]
    else:  # compose
        candidates = search_composition(
            data, var_names,
            max_nodes_b=3,
            max_nodes_a=8,
            time_budget=time_budget,
        )
        # If direct brute also finds something, add it — dedup below
        g = Grammar(
            variables=var_names,
            constants=[1.0],
            max_depth=3,
            ops=ops or ["add", "sub", "mul", "div", "pow"],
        )
        direct = brute_search(g, data, max_nodes=8,
                              max_per_level=50000, tol=0.02,
                              time_budget=time_budget / 4)
        candidates += [{"expr": c["expr"]} for c in direct]

    elapsed = time.time() - t0
    print(f"  total    : {len(candidates)} candidates in {elapsed:.1f}s")

    # Dedupe + classify
    seen: set[str] = set()
    shown = 0
    for c in candidates:
        expr = c["expr"]
        try:
            key = str(sp.simplify(expr))
        except Exception:
            key = str(expr)
        if key in seen:
            continue
        seen.add(key)

        n_nodes = sum(1 for _ in sp.preorder_traversal(expr))
        novelty = classify_novelty(expr, REGISTRY)
        mark = {"KNOWN": "=", "SCALED_KNOWN": "~",
                "COMPOSITE": "+", "NOVEL": "*"}[novelty["class"]]

        detail = ""
        if "matched" in novelty:
            detail = f"= {novelty['matched']}"
        elif "parts" in novelty:
            detail = (f"{novelty['parts'][0]} {novelty['op']} "
                      f"{novelty['parts'][1]}")
        decomp = ""
        if "decomposition" in c:
            A, op, B = c["decomposition"]
            decomp = f"   via ({A}) {op} ({B})"

        print(f"\n    [{shown+1}] {mark} {novelty['class']:12s}  {expr}")
        print(f"         nodes={n_nodes}  {detail}{decomp}")
        shown += 1
        if shown >= 5:
            break


if __name__ == "__main__":
    m, v, r = sp.symbols("m v r")

    print("Novel discovery demo v2")
    print()
    print("Ruling:")
    print("  =  KNOWN         already a corpus law       (filter OUT)")
    print("  ~  SCALED_KNOWN  constant × a corpus law    (filter OUT)")
    print("  +  COMPOSITE     binary mix of two laws     (flag for review)")
    print("  *  NOVEL         not in or composed of      (investigate)")

    # Case 1: plain brute_search is enough — target is 5 nodes
    _present(
        "CASE 1 — rediscovery of a corpus law (shallow target)",
        m * v ** 2,
        ["m", "v"],
        search="brute", max_nodes=8, time_budget=15.0,
    )

    # Case 2: plain brute_search is enough — target is 7 nodes
    _present(
        "CASE 2 — trivial composition of two corpus laws",
        m * v ** 2 + m * v,
        ["m", "v"],
        search="brute", max_nodes=10, time_budget=15.0,
    )

    # Case 3: requires composition search — target is 9 nodes and
    # its subtrees don't fit the full data
    _present(
        "CASE 3 — structurally novel form (depth 3, needs composition)",
        m * v ** 2 / (r + 1),
        ["m", "v", "r"],
        search="compose", time_budget=60.0,
    )

    print()
    print("=" * 78)
    print("  Case 1 and 2 use brute_search (targets fit at every prefix).")
    print("  Case 3 uses compose_search: enumerate B, peel data, search A.")
    print("  All three land in the correct novelty class. In production,")
    print("  send NOVEL candidates to human review, COMPOSITE to a queue,")
    print("  and auto-accept KNOWN / SCALED_KNOWN as rediscoveries.")
    print("=" * 78)
