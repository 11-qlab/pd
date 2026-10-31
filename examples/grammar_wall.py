"""Demonstrate three levels of grammar restriction and their effects.

- Full grammar:           multiplication is reachable directly
- No-mul (leaks):         multiplication reachable via  a/(1/m)  and  (a+m)^2
- Add-only (true wall):   multiplication is genuinely unreachable
"""
import random
from purediscover import Grammar, discover_law

rng = random.Random(7)
data = []
for _ in range(40):
    m = rng.uniform(0.5, 5.0)
    a = rng.uniform(0.5, 5.0)
    data.append(({"m": m, "a": a}, m * a))

for ops_label, ops in [
    ("with mul (full)",       ["add", "mul", "div", "pow"]),
    ("no mul (leaks)",        ["add", "div", "pow"]),
    ("no div, no mul",        ["add", "pow"]),
    ("add only (true wall)",  ["add"]),
]:
    g = Grammar(variables=["m", "a"], constants=[1.0, 2.0],
                max_depth=3, ops=ops)
    top = discover_law(g, data, n_candidates=3000, tol=0.05)
    if top:
        best = top[0]["expr"]
        best_bits = top[0]["score"]["bits"]
        best_res = top[0]["score"]["residual"]
        has_cross_mul = ("*" in str(best) and "mul" not in ops and "a" in str(best) and "m" in str(best))
        tag = "  ⚠ leak" if has_cross_mul else ""
        print(f"  ops={ops_label:24s}  best={str(best):20s}  "
              f"bits={best_bits:6.2f}  residual={best_res:.4f}{tag}")
    else:
        print(f"  ops={ops_label:24s}  (no candidate passed the gate)")
