"""Rediscovers the SHAPE m1*m2/r^2. G is scaled out (it's a units constant,
not a structural discovery)."""
import random
from purediscover import Grammar, discover_law

rng = random.Random(11)
data = []
for _ in range(60):
    m1 = rng.uniform(1.0, 10.0)
    m2 = rng.uniform(1.0, 10.0)
    r = rng.uniform(1.0, 5.0)
    # target = m1*m2/r^2 (G factored out; units = kg^2 / m^2)
    data.append(({"m1": m1, "m2": m2, "r": r}, m1 * m2 / r ** 2))

g = Grammar(variables=["m1", "m2", "r"], constants=[1.0, 2.0], max_depth=3)
top = discover_law(g, data, n_candidates=5000, tol=0.05)

print("Top candidates (target: m1*m2/r^2):")
for r in top[:5]:
    print(f"  {r['expr']}  bits={r['score']['bits']:.1f}  size={r['score']['size']}")
