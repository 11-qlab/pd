"""Same as newton.py but with 5% noise — the realistic case."""
import random
from purediscover import Grammar, discover_law, replay_audit

rng = random.Random(7)
data = []
for _ in range(40):
    m = rng.uniform(0.5, 5.0)
    a = rng.uniform(0.5, 5.0)
    # 5% noise added to the target
    data.append(({"m": m, "a": a}, m * a + rng.gauss(0, 0.05 * m * a)))

g = Grammar(variables=["m", "a"], constants=[1.0, 2.0], max_depth=3)
top = discover_law(g, data, n_candidates=3000, tol=0.10)

print("Top candidates (target: a*m within 5% noise):")
for r in top[:5]:
    print(f"  {r['expr']}  bits={r['score']['bits']:.1f}  "
          f"residual={r['score']['residual']:.4f}")

held = []
for _ in range(20):
    m = rng.uniform(0.5, 5.0)
    a = rng.uniform(0.5, 5.0)
    held.append(({"m": m, "a": a}, m * a + rng.gauss(0, 0.05 * m * a)))
if top:
    print("\nAudit on held-out:")
    print(" ", replay_audit(top[0]["expr"], held, tolerance=0.15))
