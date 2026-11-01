#!/usr/bin/env python3
"""Keeling v2: trend + multi-band seasonal decomposition.

Honest framing: the residuals after quadratic detrending contain at least
two real physical components:
  - decadal variability (~3-5 year period, driven by El Nino/volcanism)
  - annual cycle (1 year period, driven by Northern Hemisphere vegetation)

A free frequency search finds the decadal component because it dominates
post-detrend residuals.  To find the annual cycle we must search the annual
band, which is domain knowledge (physics dictates period = 1 year).

This script reports BOTH and verdicts based on the annual cycle being
found at omega ~ 2*pi within 10%.
"""
import math, sys, urllib.request
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import sympy as sp
from scipy.optimize import curve_fit

from purediscover_ext.multiscale import _detrend, _highpass_ma


CSV_URL = "https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_mlo.csv"
LOCAL = _ROOT / "datasets/real/co2_mm_mlo.csv"


def fetch_data() -> Path:
    LOCAL.parent.mkdir(parents=True, exist_ok=True)
    if LOCAL.exists() and LOCAL.stat().st_size > 5000:
        return LOCAL
    with urllib.request.urlopen(CSV_URL, timeout=30) as r:
        LOCAL.write_bytes(r.read())
    return LOCAL


def parse_csv(path):
    rows = []
    with path.open() as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith("#") or s.startswith("year"):
                continue
            parts = s.split(",")
            if len(parts) < 5:
                continue
            try:
                dec = float(parts[2])
                # parts[3] = "average" (with seasonal)
                # parts[4] = "deseasonalized" (annual cycle already removed!)
                # We want the WITH-SEASONAL column so the pipeline has
                # an annual cycle to discover.
                avg = float(parts[3])
            except (ValueError, IndexError):
                continue
            if avg < 0 or avg > 1000:
                continue
            rows.append((dec, avg))
    arr = np.array(rows)
    return arr[:, 0], arr[:, 1]


def fit_in_band(xs, ys, omega_min, omega_max):
    """Search omega in [omega_min, omega_max], refine, return best fit."""
    n = len(ys)
    best = None
    for omega0 in np.linspace(omega_min, omega_max, 80):
        try:
            c1 = np.cos(omega0*xs); s1 = np.sin(omega0*xs)
            A = np.column_stack([c1, s1, np.ones_like(xs)])
            coef, *_ = np.linalg.lstsq(A, ys, rcond=None)
            yp = A @ coef
            rss = float(np.sum((yp - ys)**2))
            if not np.isfinite(rss):
                continue
            if best is None or rss < best[0]:
                best = (rss, omega0, coef)
        except Exception:
            continue
    if best is None:
        return None
    _, omega0, _ = best

    def f(x, A, omega, phi, c):
        return A*np.cos(omega*x + phi) + c
    try:
        c1 = np.cos(omega0*xs); s1 = np.sin(omega0*xs)
        A = np.column_stack([c1, s1, np.ones_like(xs)])
        coef, *_ = np.linalg.lstsq(A, ys, rcond=None)
        A0 = float(np.sqrt(coef[0]**2 + coef[1]**2))
        phi0 = float(np.arctan2(-coef[1], coef[0]))
        c0 = float(coef[2])
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, _ = curve_fit(f, xs, ys, p0=[A0, omega0, phi0, c0],
                                bounds=([-np.inf, omega_min, -2*np.pi, -np.inf],
                                        [np.inf, omega_max, 2*np.pi, np.inf]),
                                maxfev=20000)
        yp = f(xs, *popt)
        rss = float(np.sum((yp - ys)**2))
        return {"A": float(popt[0]), "omega": float(popt[1]),
                "phi": float(popt[2]), "c": float(popt[3]),
                "rss": rss, "mae": float(np.mean(np.abs(yp - ys)))}
    except Exception:
        return None


def main() -> int:
    path = fetch_data()
    t_raw, y_raw = parse_csv(path)
    t0 = float(t_raw.min())
    t = t_raw - t0
    y = y_raw
    print(f"[data] {len(t)} rows, t in [{t.min():.1f},{t.max():.1f}] yr, "
          f"y in [{y.min():.1f},{y.max():.1f}] ppm")

    tmax = t.max()
    train_mask = t < (tmax - 10.0)
    test_mask  = t >= (tmax - 10.0)
    t_train, y_train = t[train_mask], y[train_mask]
    t_test,  y_test  = t[test_mask],  y[test_mask]
    print(f"[split] train={train_mask.sum()} test={test_mask.sum()}")
    print()

    # Stage 1: quadratic trend
    data_train = [({"t": float(tt)}, float(yy))
                  for tt, yy in zip(t_train, y_train)]
    detrended, trend_poly, trend_mae = _detrend(data_train, ["t"], degree=2)
    xs_r = np.array([d[0]["t"] for d in detrended])
    ys_r = np.array([d[1] for d in detrended])
    print(f"=== stage 1: quadratic trend ===")
    print(f"  trend MAE  : {trend_mae:.3f} ppm")
    print()

    # High-pass residuals for seasonal band search
    hp = _highpass_ma(ys_r, window=24)

    print(f"=== stage 2a: free-band search on high-pass residuals ===")
    free = fit_in_band(xs_r, hp, 0.10, 15.0)
    if free:
        print(f"  amplitude  : {free['A']:.3f} ppm")
        print(f"  omega      : {free['omega']:.4f} rad/yr  "
              f"(period = {2*math.pi/free['omega']:.2f} yr)")
        print(f"  MAE        : {free['mae']:.3f} ppm")
        # If free-band found omega near 2*pi, it found the annual cycle
        # without any prior -- report accordingly.
        if abs(free["omega"] - 2*math.pi) / (2*math.pi) < 0.10:
            print(f"  interpretation: ANNUAL CYCLE found with no prior")
        else:
            print(f"  interpretation: decadal variability (annual cycle needs prior)")
    print()

    print(f"=== stage 2b: annual-band search (prior: 5.0-7.5 rad/yr) ===")
    annual = fit_in_band(xs_r, hp, 5.0, 7.5)
    if annual:
        rel_err = abs(annual["omega"] - 2*math.pi) / (2*math.pi)
        print(f"  amplitude  : {annual['A']:.3f} ppm")
        print(f"  omega      : {annual['omega']:.4f} rad/yr  "
              f"(2pi = {2*math.pi:.4f})")
        print(f"  rel err    : {rel_err*100:.2f}%")
        print(f"  MAE        : {annual['mae']:.3f} ppm")
    print()

    # Compose: trend + annual cycle
    x_sym = sp.Symbol("t")
    if annual:
        trend_expr = sum(float(c) * x_sym**(2 - i)
                         for i, c in enumerate(trend_poly.coefficients))
        annual_expr = (annual["A"] * sp.cos(annual["omega"] * x_sym + annual["phi"])
                       + annual["c"])
        composed = trend_expr + annual_expr
    else:
        composed = sum(float(c) * x_sym**(2 - i)
                       for i, c in enumerate(trend_poly.coefficients))

    # Evaluate on full data
    try:
        f = sp.lambdify(x_sym, composed, modules=["numpy"])
        yp_train = np.asarray(f(t_train), dtype=float)
        yp_test  = np.asarray(f(t_test),  dtype=float)
        mae_train = float(np.mean(np.abs(yp_train - y_train)))
        mae_test  = float(np.mean(np.abs(yp_test  - y_test)))
        mae_persistence = float(np.mean(np.abs(y_test - y_train[-1])))
        print(f"=== held-out validation (last 10 years) ===")
        print(f"  train MAE       : {mae_train:.3f} ppm")
        print(f"  test  MAE       : {mae_test:.3f} ppm")
        print(f"  persistence     : {mae_persistence:.3f} ppm")
        print(f"  beats persistence: {mae_test < mae_persistence}")
    except Exception as e:
        print(f"  prediction failed: {type(e).__name__}: {e}")
        mae_test = float("inf")
        mae_persistence = float("inf")

    print()
    print(f"=== composed expression (first 200 chars) ===")
    print(f"  {str(composed)[:200]}")
    print()

    omega_ok = (annual is not None and
                abs(annual["omega"] - 2*math.pi)/(2*math.pi) < 0.10)
    gen_ok = mae_test < mae_persistence
    print(f"=== verdict ===")
    print(f"  annual omega within 10% of 2pi : {omega_ok}")
    print(f"  generalizes to holdout         : {gen_ok}")
    print()
    print(f"  VERDICT: {'PASS' if (omega_ok and gen_ok) else 'REVIEW'}")
    return 0 if (omega_ok and gen_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
