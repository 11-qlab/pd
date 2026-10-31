"""Trace exactly why the 3 fixes aren't firing."""
import math
import random
import sympy as sp
import numpy as np

print("=" * 66)
print("BUG 1: _laurent_offset on 2*sqrt(x) — where does it pass?")
print("=" * 66)

# Reproduce verify_offset.py's exact input
rng = random.Random(3)
xs, ys = [], []
for _ in range(30):
    x = rng.uniform(0.5, 10.0)
    y = (2.0 * math.sqrt(x)) * (1.0 + rng.gauss(0.0, 0.02))
    xs.append(x); ys.append(y)

x_arr = np.array(xs); y_arr = np.array(ys)
X = np.column_stack([np.ones(len(xs)), 1.0/x_arr, 1.0/(x_arr**2)])
coefs, *_ = np.linalg.lstsq(X, y_arr, rcond=None)
y_pred = X @ coefs
resid = float(np.mean(np.abs(y_arr - y_pred) / (np.abs(y_arr) + 1e-9)))
print(f"  full fit c0={coefs[0]:+.4f}  resid={resid:.4f}")

# Endpoint
n_top = max(3, len(x_arr) // 10)
top_idx = np.argsort(x_arr)[-n_top:]
resid_top = float(np.mean(
    np.abs(y_arr[top_idx] - y_pred[top_idx])
    / (np.abs(y_arr[top_idx]) + 1e-9)
))
print(f"  endpoint resid_top = {resid_top:.4f}  (>0.08 rejects)")
print(f"  endpoint x samples: {x_arr[top_idx]}")
print(f"  endpoint y actual : {y_arr[top_idx]}")
print(f"  endpoint y pred   : {y_pred[top_idx]}")

print()
print("=" * 66)
print("BUG 2: _snap_coefficients on Kepler form")
print("=" * 66)

from purediscover.discover import _snap_coefficients, _snap_float, _simplify, _denormalize

test_expr = sp.Float(1.0) * sp.Symbol('a') ** sp.Rational(1309, 873)
print(f"  input         : {test_expr}")
print(f"  atoms Float   : {list(test_expr.atoms(sp.Float))}")
print(f"  atoms Rational: {list(test_expr.atoms(sp.Rational))}")
print()
print(f"  _snap_float(1.0)  = {_snap_float(1.0)}")
print(f"  _snap_float(1.4995)= {_snap_float(1309/873)}")
print(f"  _snap_coefficients(input) = {_snap_coefficients(test_expr)}")
print(f"  _simplify(input)          = {_simplify(test_expr)}")
print(f"  _denormalize(input, 1.0)  = {_denormalize(test_expr, 1.0)}")

# Try with y_scale slightly off 1.0
print()
print(f"  _denormalize(a**(1309/873), 1.0003) = {_denormalize(sp.Symbol('a')**sp.Rational(1309,873), 1.0003)}")

print()
print("=" * 66)
print("BUG 3: search_offset on demo CSV — how loose is brute_search tol?")
print("=" * 66)

import csv
data = []
with open("datasets/input/novel_ax_b.csv", newline="") as f:
    for row in csv.DictReader(f):
        data.append(({"x": float(row["x"])}, float(row["y"])))

from purediscover.compose import _laurent_offset, _fit_error
B = _laurent_offset(data, ["x"])
print(f"  B = {B}")

if B is not None:
    residuals = [abs(y - B) for _, y in data]
    residual_scale = max(residuals)
    transformed = [(env, (y - B) / residual_scale) for env, y in data]
    print(f"  transformed y range: {min(t for _, t in transformed):.4f} "
          f".. {max(t for _, t in transformed):.4f}")

    # Try brute_search at multiple tolerances
    from purediscover.symbolic import Grammar
    from purediscover.rich_search import brute_search
    for stol in (0.09, 0.2, 0.4, 0.6):
        g = Grammar(variables=["x"], constants=[1.0], max_depth=3,
                    ops=["add", "sub", "mul", "div", "pow", "sqrt", "log", "exp"])
        cands = brute_search(g, transformed, max_nodes=8,
                             max_per_level=50000, tol=stol,
                             time_budget=5.0)
        print(f"  tol={stol:>4}  -> {len(cands)} candidates")
        if cands and stol >= 0.2:
            print(f"    sample: {cands[0]['expr']}")
