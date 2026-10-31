"""Explore small groups against a physics-inspired scoring function."""
from purediscover import explore_groups


def score(c):
    # Reward non-abelian + small order. Higher = better.
    return (0.0 if c.abelian else 1.0) + 0.1 * (1.0 / c.order)


top = explore_groups(score, max_order=12, top_k=5)
print("Top groups by score (smallest non-abelian first):")
for c in top:
    print(f"  {c.name:>6}  order={c.order:>3}  abelian={c.abelian}  "
          f"score={c.score:.3f}  ({c.description})")
