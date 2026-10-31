"""Rediscovers F = m*a from (m, a, F) data."""
import random
from purediscover import Grammar, discover_law, replay_audit
import sympy as sp

rng = random.Random(7)
data = []
for _ in range(40):
    m = rng.uniform(0.5, 5.0)
    a = rng.uniform(0.5, 5.0)
    data.append(({"m": m, "a": a}, m * a))

g = Grammar(variables=["m", "a"], constants=[1.0, 2.0], max_depth=3)
top = discover_law(g, data, n_candidates=3000)

print("Top candidates by MDL bits:")
for r in top[:5]:
    print(f"  {r['expr']}  bits={r['score']['bits']:.1f}  size={r['score']['size']}")

# Replay audit on held-out
held = []
for _ in range(20):
    m = rng.uniform(0.5, 5.0)
    a = rng.uniform(0.5, 5.0)
    held.append(({"m": m, "a": a}, m * a))
if top:
    print("\nAudit of best candidate:")
    print(" ", replay_audit(top[0]["expr"], held))
