"""Trace search_offset step by step on novel data."""
import csv
import math
import pathlib
import random

import sympy as sp

from purediscover.compose import _laurent_offset, _fit_error, search_offset


# Regenerate CSV with multiplicative noise (same as fixed demo)
path = pathlib.Path("datasets/input/novel_ax_b.csv")
rng = random.Random(3)
log_lo = math.log(0.2)
log_hi = math.log(20.0)
with path.open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["x", "y"])
    for _ in range(30):
        # Log-uniform: ~1 sample per log-decade, matches demo
        x = math.exp(rng.uniform(log_lo, log_hi))
        y = (2.0 / x + 1.0) * (1.0 + rng.gauss(0.0, 0.02))
        w.writerow([f"{x:.6f}", f"{y:.6f}"])

data = []
with path.open(newline="") as f:
    r = csv.DictReader(f)
    for row in r:
        data.append(({"x": float(row["x"])}, float(row["y"])))

print(f"Loaded {len(data)} rows")
print(f"  y range: {min(y for _, y in data):.3f} .. {max(y for _, y in data):.3f}")

# Step 1: Laurent offset
B = _laurent_offset(data, ["x"])
print(f"\nStep 1: _laurent_offset -> B = {B}")

if B is not None:
    # Step 2: transformed data
    transformed = [(env, y - B) for env, y in data]
    t_range = (min(y for _, y in transformed), max(y for _, y in transformed))
    print(f"Step 2: transformed y range = {t_range[0]:.3f} .. {t_range[1]:.3f}")

    # Step 3: check fit error of reconstructed target
    target = sp.sympify("2.0/x + 1.0")
    fe = _fit_error(target, data)
    print(f"Step 3: fit_error(true target '2/x + 1') = {fe:.5f}")
    # Check fit error of just 2/x (no offset)
    fe2 = _fit_error(sp.sympify("2.0/x"), data)
    print(f"         fit_error('2/x' no offset)       = {fe2:.5f}")

    # Step 4: run search_offset
    print(f"\nStep 4: search_offset(tol=0.03, budget=15s)")
    results = search_offset(data, ["x"], tol=0.03, time_budget=15.0)
    print(f"  -> {len(results)} candidates")
    for i, r in enumerate(results[:3], 1):
        print(f"    {i}. {r['expr']}   fit={r['fit_error']:.5f}  n={r['n_nodes']}")
