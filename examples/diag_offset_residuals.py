"""See exact Laurent residuals for the failing cases."""
import math
import random
import sympy as sp

from purediscover.compose import _laurent_offset


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


def _report(label, fn, x_range, noise):
    data = _gen(fn, x_range, noise=noise, seed=0)
    # Inline the Laurent computation to expose residual
    import numpy as np
    xs = np.array([e["x"] for e, _ in data])
    ys = np.array([y for _, y in data])
    X = np.column_stack([np.ones(len(xs)), 1.0/xs, 1.0/(xs**2)])
    coefs, *_ = np.linalg.lstsq(X, ys, rcond=None)
    y_pred = X @ coefs
    resid = float(np.mean(np.abs(ys - y_pred) / (np.abs(ys) + 1e-9)))

    print(f"\n  {label}")
    print(f"    fn        : {fn}")
    print(f"    x range   : {x_range}  noise: {noise*100:.0f}%")
    print(f"    coefs     : c0={coefs[0]:+.4f}  c1={coefs[1]:+.4f}  c2={coefs[2]:+.4f}")
    print(f"    resid     : {resid:.4f}")
    print(f"    returned B: {_laurent_offset(data, ['x'])}")


print("Laurent residual diagnostic")
print("=" * 66)

_report("y = 2/x + 1  (offset present, no noise)",
        lambda x: 2.0/x + 1.0, (0.2, 20.0), 0.0)
_report("y = 2/x + 1  (offset present, 2% noise)",
        lambda x: 2.0/x + 1.0, (0.2, 20.0), 0.02)
_report("y = 5/x  (no offset)",
        lambda x: 5.0/x, (0.5, 10.0), 0.02)
_report("y = 2*sqrt(x)  (no offset, WRONG)",
        lambda x: 2.0*math.sqrt(x), (0.5, 10.0), 0.02)
_report("y = 2*sqrt(x)  (no offset, WRONG, wider x)",
        lambda x: 2.0*math.sqrt(x), (0.2, 20.0), 0.02)
_report("y = exp(-x)  (no offset, incompatible)",
        lambda x: math.exp(-x), (0.5, 10.0), 0.02)
