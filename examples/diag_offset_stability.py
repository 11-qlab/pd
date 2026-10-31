"""Trace the stability check on each test case."""
import math
import random
import numpy as np

from purediscover.compose import _laurent_offset


def _gen(fn, x_range, n=30, noise=0.0, seed=0, log=False):
    rng = random.Random(seed)
    lo, hi = x_range
    xs, ys = [], []
    for _ in range(n):
        if log:
            x = math.exp(rng.uniform(math.log(lo), math.log(hi)))
        else:
            x = rng.uniform(lo, hi)
        y = fn(x) * (1.0 + rng.gauss(0, noise))
        xs.append(x); ys.append(y)
    return [({"x": x}, y) for x, y in zip(xs, ys)]


def _trace(label, data, expect):
    xs = np.array([e["x"] for e, _ in data])
    ys = np.array([y for _, y in data])
    X = np.column_stack([np.ones(len(xs)), 1.0/xs, 1.0/(xs**2)])
    coefs, *_ = np.linalg.lstsq(X, ys, rcond=None)
    y_pred = X @ coefs
    resid = float(np.mean(np.abs(ys - y_pred) / (np.abs(ys) + 1e-9)))
    x_min, x_max = xs.min(), xs.max()
    x_mid = math.sqrt(x_min * x_max)
    m_lo = xs <= x_mid
    m_hi = xs > x_mid
    B_lo = B_hi = None
    if m_lo.sum() >= 5 and m_hi.sum() >= 5:
        B_lo = float(np.linalg.lstsq(X[m_lo], ys[m_lo], rcond=None)[0][0])
        B_hi = float(np.linalg.lstsq(X[m_hi], ys[m_hi], rcond=None)[0][0])

    B = _laurent_offset(data, ["x"])
    print(f"\n  {label}")
    print(f"    resid       : {resid:.4f}  (>0.025 rejects)")
    print(f"    c0 (full)   : {coefs[0]:+.4f}")
    print(f"    B_lo (<x_mid): {B_lo:+.4f}" if B_lo is not None else "    B_lo        : (skip)")
    print(f"    B_hi (>x_mid): {B_hi:+.4f}" if B_hi is not None else "    B_hi        : (skip)")
    if B_lo is not None and B_hi is not None:
        scale = max(abs(B_lo), abs(B_hi), 1e-9)
        diff = abs(B_lo - B_hi)
        print(f"    |B_lo-B_hi|: {diff:.4f}  (>0.5*scale rejects; scale={scale:.4f})")
    print(f"    returned    : {B}  (expect {expect})")


print("Stability check diagnostic")
print("=" * 66)
_trace("2/x + 1  log-sample wide",
       _gen(lambda x: 2/x + 1, (0.2, 20), noise=0.02, seed=3, log=True),
       "≈ 1")
_trace("2/x + 1  uniform narrow",
       _gen(lambda x: 2/x + 1, (0.5, 10), noise=0.02, seed=7),
       "≈ 1")
_trace("2*sqrt(x)  uniform",
       _gen(lambda x: 2*math.sqrt(x), (0.5, 10), noise=0.02, seed=3),
       "None")
_trace("5/x  uniform",
       _gen(lambda x: 5/x, (0.5, 10), noise=0.02, seed=2),
       "near 0")
_trace("exp(-x)  uniform",
       _gen(lambda x: math.exp(-x), (0.5, 10), noise=0.02, seed=4),
       "None")
