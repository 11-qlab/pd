"""Multi-scale discovery: separate slow (trend) from fast (seasonal) components.

Stage 1: fit trend on smoothed/downsampled data -> T(t)
Stage 2: residuals r = y - T(t) -> discover seasonal S(t) on r
Combine: y_pred = T(t) + S(t)

This is the missing preprocessing stage that Keeling v1 exposed.
"""
from __future__ import annotations

import math
import warnings

import numpy as np
import sympy as sp
from scipy.optimize import curve_fit


def _poly_fit(x, y, degree=2):
    coef = np.polyfit(x, y, degree)
    return np.poly1d(coef)


def _detrend(data, var_names, degree=2):
    """Fit polynomial trend, return (detrended_data, trend_coefs, trend_mae)."""
    xn = var_names[0]
    xs = np.array([d[0][xn] for d in data], dtype=float)
    ys = np.array([d[1] for d in data], dtype=float)
    p = _poly_fit(xs, ys, degree)
    trend = p(xs)
    resid = ys - trend
    detrended = [({xn: float(xs[i])}, float(resid[i]))
                 for i in range(len(resid))]
    trend_mae = float(np.mean(np.abs(trend - ys)))
    return detrended, p, trend_mae


def _discover_seasonal(detrended_data, var_names):
    """Fit cos(a*t + b) to detrended series via grid+refine."""
    xn = var_names[0]
    xs = np.array([d[0][xn] for d in detrended_data], dtype=float)
    ys = np.array([d[1] for d in detrended_data], dtype=float)
    n = len(ys)
    x_sym = sp.Symbol(xn)

    # grid over plausible annual frequencies (2π/year ± range)
    best = None
    for omega0 in np.linspace(0.1, 15.0, 300):
        try:
            c1 = np.cos(omega0*xs); s1 = np.sin(omega0*xs)
            A = np.column_stack([c1, s1, np.ones_like(xs)])
            coef, *_ = np.linalg.lstsq(A, ys, rcond=None)
            yp = A @ coef
            rss = float(np.sum((yp - ys)**2))
            if not np.isfinite(rss):
                continue
            if best is None or rss < best[0]:
                best = (rss, float(omega0), coef)
        except Exception:
            continue

    if best is None:
        return None
    rss, omega0, _ = best

    # refine
    def f(x, A, omega, phi, c):
        return A*np.cos(omega*x + phi) + c
    try:
        c1 = np.cos(omega0*xs); s1 = np.sin(omega0*xs)
        A = np.column_stack([c1, s1, np.ones_like(xs)])
        coef, *_ = np.linalg.lstsq(A, ys, rcond=None)
        A0 = float(np.sqrt(coef[0]**2 + coef[1]**2))
        phi0 = float(np.arctan2(-coef[1], coef[0]))
        c0 = float(coef[2])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, _ = curve_fit(f, xs, ys, p0=[A0, omega0, phi0, c0],
                                bounds=([-np.inf, 0.01, -2*np.pi, -np.inf],
                                        [np.inf, 100.0, 2*np.pi, np.inf]),
                                maxfev=20000)
        expr = (popt[0]*sp.cos(popt[1]*x_sym + popt[2]) + popt[3])
        yp = f(xs, *popt)
        mae = float(np.mean(np.abs(yp - ys)))
        return {"expr": expr, "expr_str": str(expr),
                "params": [float(p) for p in popt],
                "mae": mae, "rss": float(np.sum((yp-ys)**2))}
    except Exception:
        return None


def _highpass_ma(ys, window=24):
    """Subtract moving average to remove slow components (>= window period)."""
    if len(ys) < window:
        return ys - np.mean(ys)
    kernel = np.ones(window) / window
    smooth = np.convolve(ys, kernel, mode='same')
    edge = window // 2
    smooth[:edge] = smooth[edge]
    smooth[-edge:] = smooth[-edge - 1]
    return ys - smooth


def discover_multiscale(data, var_names, *, trend_degree=2, highpass_window=24):
    """Two-stage: trend + seasonal."""
    if len(var_names) != 1:
        raise ValueError(f"multiscale needs 1 var, got {var_names}")
    xn = var_names[0]
    xs = np.array([d[0][xn] for d in data], dtype=float)
    ys = np.array([d[1] for d in data], dtype=float)

    detrended, trend_poly, trend_mae = _detrend(data, var_names, trend_degree)
    # high-pass residuals: removes slow drift, isolates annual-scale components
    ys_r = np.array([d[1] for d in detrended], dtype=float)
    hp = _highpass_ma(ys_r, highpass_window)
    hp_data = [({xn: detrended[i][0][xn]}, float(hp[i]))
               for i in range(len(detrended))]
    seasonal = _discover_seasonal(hp_data, var_names)

    if seasonal is None:
        return {"trend_mae": trend_mae, "seasonal": None,
                "composed_expr": "?",
                "final_mae": trend_mae,
                "trend_poly": trend_poly}

    x_sym = sp.Symbol(xn)
    trend_expr = sum(float(c) * x_sym**(trend_degree - i)
                     for i, c in enumerate(trend_poly.coefficients))
    composed = trend_expr + seasonal["expr"]
    try:
        f = sp.lambdify(x_sym, composed, modules=["numpy"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            yp = np.asarray(f(xs), dtype=float)
        final_mae = float(np.mean(np.abs(yp - ys)))
    except Exception:
        final_mae = float("inf")

    return {
        "trend_mae": trend_mae,
        "seasonal": seasonal,
        "composed_expr": str(composed),
        "final_mae": final_mae,
        "trend_poly": trend_poly,
    }
