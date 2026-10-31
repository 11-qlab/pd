"""Probe: which unary ops does brute_search's dispatcher actually support?

Creates a 1-var Grammar with a single op, feeds data from a known target
expression, and reports whether brute_search recovers it.
"""
import random
import sympy as sp

from purediscover.symbolic import Grammar
from purediscover.rich_search import brute_search


def probe(op: str, target_builder, var: str = "x",
          max_nodes: int = 5, time_budget: float = 8.0):
    g = Grammar(variables=[var], constants=[1.0], max_depth=4, ops=[op])
    x = sp.Symbol(var)
    target = target_builder(x)
    rng = random.Random(0)
    data = []
    for _ in range(30):
        xv = rng.uniform(0.5, 3.0)
        try:
            y = float(target.subs({x: xv}))
        except Exception:
            return (op, "💥", "target eval failed")
        if not (abs(y) < 1e12):
            continue
        data.append(({var: xv}, y))
    try:
        cands = brute_search(g, data, max_nodes=max_nodes,
                             max_per_level=20000, tol=0.02,
                             time_budget=time_budget)
    except Exception as e:
        return (op, "💥", f"{type(e).__name__}: {e}")
    if cands:
        return (op, "✓", str(cands[0]["expr"]))
    return (op, "✗", "0 candidates")


print("PROBE: brute_search dispatcher op support")
print("=" * 66)
probes = [
    ("sqrt",  lambda x: sp.sqrt(x)),
    ("log",   lambda x: sp.log(x)),
    ("exp",   lambda x: sp.exp(x)),
    ("pow",   lambda x: x ** 2),
    ("pow",   lambda x: x ** 4),                # exponent-4 support check
    ("sin",   lambda x: sp.sin(x)),
    ("tan",   lambda x: sp.tan(x)),
    ("abs",   lambda x: sp.Abs(x)),
    ("inv",   lambda x: 1 / x),
    ("neg",   lambda x: -x),
]
for op, builder in probes:
    name, verdict, detail = probe(op, builder)
    print(f"  {name:5s} -> {detail[:20]:20s}  {verdict}  {detail[20:] if len(detail)>20 else ''}")
print("=" * 66)
print()
print("Read: rows marked '✓' mean the dispatcher handles that op.")
print("      '✗ 0 candidates' means brute_search has no emission path.")
print()
print("If sqrt=✗, then pendulum_period and hypotenuse need")
print("  a patch to rich_search.py to emit pow(x, 1/2), or")
print("  acceptance as known misses.")
