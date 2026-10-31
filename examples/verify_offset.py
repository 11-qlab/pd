"""Verify offset detection on 4 data types."""
import math
import random

import sympy as sp

from purediscover.compose import _laurent_offset, search_offset


def _gen(fn, x_range, n=30, noise=0.0, seed=0):
    rng = random.Random(seed)
    lo, hi = x_range
    out = []
    for _ in range(n):
        x = rng.uniform(lo, hi)
        y = fn(x)
        if noise > 0:
            y *= (1.0 + rng.gauss(0.0, noise))
        out.append(({"x": x}, y))
    return out


def _row(label, data, expect_B):
    B = _laurent_offset(data, ["x"])
    if B is None:
        print(f"  {label:32s}  offset=None (incompatible)")
        return
    ok = "✓" if expect_B is None or abs(B - expect_B) < 0.05 * max(1.0, abs(expect_B)) else "✗"
    print(f"  {label:32s}  offset={B:+.4f}  expected≈{expect_B if expect_B is not None else '?'}  {ok}")


print("Laurent offset detection")
print("=" * 66)
_row("y = 2/x + 1  (offset present)",
     _gen(lambda x: 2.0/x + 1.0, (0.5, 10.0), noise=0.02), 1.0)
_row("y = 3/x + 0.5  (offset present)",
     _gen(lambda x: 3.0/x + 0.5, (0.5, 10.0), noise=0.02, seed=1), 0.5)
_row("y = 5/x  (no offset)",
     _gen(lambda x: 5.0/x, (0.5, 10.0), noise=0.02, seed=2), 0.0)
_row("y = 2*sqrt(x)  (no offset)",
     _gen(lambda x: 2.0*math.sqrt(x), (0.5, 10.0), noise=0.02, seed=3), 0.0)
_row("y = exp(-x)  (no offset, not Laurent)",
     _gen(lambda x: math.exp(-x), (0.5, 10.0), noise=0.02, seed=4), None)


def _gen_log(fn, x_range, n=30, noise=0.0, seed=0):
    """Log-uniform sampling — required for 1/x-family laws to be visible."""
    rng = random.Random(seed)
    lo, hi = x_range
    out = []
    for _ in range(n):
        x = math.exp(rng.uniform(math.log(lo), math.log(hi)))
        y = fn(x)
        if noise > 0:
            y *= (1.0 + rng.gauss(0.0, noise))
        out.append(({"x": x}, y))
    return out


print()
print("search_offset on novel_ax_b data (log-uniform, wide range)")
print("=" * 66)
data = _gen_log(lambda x: 2.0/x + 1.0, (0.2, 20.0), noise=0.02, seed=7)
results = search_offset(data, ["x"], tol=0.03, time_budget=15.0)
if results:
    for i, r in enumerate(results[:3], 1):
        print(f"  {i}. {r['expr']}   fit={r['fit_error']:.5f}  n={r['n_nodes']}")
else:
    print("  (no offset candidates)")

print()
print("search_offset on pure A/x data (should be empty)")
print("=" * 66)
data = _gen_log(lambda x: 5.0/x, (0.5, 10.0), noise=0.02, seed=8)
results = search_offset(data, ["x"], tol=0.03, time_budget=15.0)
print(f"  {len(results)} candidate(s) — expect 0")
