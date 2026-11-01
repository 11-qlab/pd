"""Recursive decomposition with FFT+grid seeding, free-omega refinement, BIC."""
from __future__ import annotations

import math
import warnings

import numpy as np
import sympy as sp
from scipy.optimize import curve_fit


def _fft_peaks(xs, ys, top_k=5):
    try:
        order = np.argsort(xs)
        x = xs[order]; y = ys[order]
        if len(y) < 8: return []
        fft = np.fft.rfft(y - np.mean(y))
        dx = (x.max() - x.min()) / (len(x) - 1)
        freqs = np.fft.rfftfreq(len(y), d=dx)
        mag = np.abs(fft); mag[0] = 0.0
        if len(mag) <= 1: return []
        idx = np.argsort(mag[1:])[-top_k:] + 1
        peaks = []
        for i in idx:
            omega = freqs[i] * 2 * np.pi
            if 0.01 < omega < 100:
                peaks.append(float(omega))
        return peaks
    except Exception:
        return []


def _grid_freq_seeds(xs, ys, omega_min=0.05, omega_max=20.0, n=80, top_k=5):
    """Linear fit at fixed omega over log grid; top-K omega by RSS."""
    if len(xs) < 5: return []
    grid = np.logspace(np.log10(omega_min), np.log10(omega_max), n)
    scored = []
    for omega in grid:
        try:
            c1 = np.cos(omega * xs); s1 = np.sin(omega * xs)
            A = np.column_stack([c1, s1, np.ones_like(xs)])
            coef, *_ = np.linalg.lstsq(A, ys, rcond=None)
            yp = A @ coef
            if not np.all(np.isfinite(yp)): continue
            rss = float(np.sum((yp - ys)**2))
            scored.append((rss, float(omega)))
        except Exception:
            continue
    scored.sort()
    return [w for _, w in scored[:top_k]]


def _fit_cos_refine(xs, ys, omega0):
    """curve_fit of A*cos(omega*t+phi)+c with omega free."""
    def f(x, A, omega, phi, c):
        return A * np.cos(omega * x + phi) + c
    try:
        c1 = np.cos(omega0 * xs); s1 = np.sin(omega0 * xs)
        A = np.column_stack([c1, s1, np.ones_like(xs)])
        coef, *_ = np.linalg.lstsq(A, ys, rcond=None)
        amp0 = float(np.sqrt(coef[0]**2 + coef[1]**2))
        phi0 = float(np.arctan2(-coef[1], coef[0]))
        c_0 = float(coef[2])
        p0 = [amp0, omega0, phi0, c_0]
        bounds = ([-np.inf, 0.01, -2*np.pi, -np.inf],
                  [np.inf, 100.0, 2*np.pi, np.inf])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, _ = curve_fit(f, xs, ys, p0=p0, bounds=bounds, maxfev=20000)
        yp = f(xs, *popt)
        if not np.all(np.isfinite(yp)): return None
        rss = float(np.sum((yp - ys)**2))
        return {'omega': float(popt[1]), 'amp': float(popt[0]),
                'phi': float(popt[2]), 'c': float(popt[3]),
                'rss': rss, 'yp': yp}
    except Exception:
        return None


def _fit_single_1d(data, var_names):
    xn = var_names[0]
    xs = np.array([d[0][xn] for d in data], dtype=float)
    ys = np.array([d[1] for d in data], dtype=float)
    n = len(ys)
    x_sym = sp.Symbol(xn)
    candidates = []

    # --- cos family: FFT + grid seeds, free-omega refinement ---
    seeds = _fft_peaks(xs, ys, top_k=5) + _grid_freq_seeds(xs, ys)
    seen_omega = set()
    for omega0 in seeds:
        key = round(omega0, 2)
        if key in seen_omega: continue
        seen_omega.add(key)
        r = _fit_cos_refine(xs, ys, omega0)
        if r is None: continue
        bic = n * np.log(r['rss']/n + 1e-15) + 4 * np.log(n)
        expr = r['amp'] * sp.cos(r['omega'] * x_sym + r['phi']) + r['c']
        candidates.append({
            'source': 'templ_cos_free',
            'expr': str(expr),
            'mae': float(np.mean(np.abs(r['yp'] - ys))),
            'bic': float(bic), 'n_params': 4, 'bits': 1e9,
            'sympy_expr': expr,
        })

    # --- library fits ---
    library = [
        ("linear", [1.0, 0.0], ([-np.inf]*2, [np.inf]*2),
         lambda x, a, b: a*x+b, 2,
         lambda p, x: p[0]*x + p[1]),
        ("quadratic", [1.0, 0.0, 0.0], ([-np.inf]*3, [np.inf]*3),
         lambda x, a, b, c: a*x**2+b*x+c, 3,
         lambda p, x: p[0]*x**2 + p[1]*x + p[2]),
        ("exp(-a*x^2)+c", [1.0, 1.0, 0.0],
         ([-np.inf, 0.001, -np.inf], [np.inf, 20.0, np.inf]),
         lambda x, a, b, c: a*np.exp(-b*x**2)+c, 3,
         lambda p, x: p[0]*sp.exp(-p[1]*x**2) + p[2]),
        ("power_law", [1.0, 1.0, 0.0],
         ([-np.inf, 0.01, -np.inf], [np.inf, 10.0, np.inf]),
         lambda x, a, b, c: a*np.power(np.abs(x), b)+c, 3,
         lambda p, x: p[0]*sp.Abs(x)**p[1] + p[2]),
    ]
    rng = np.random.default_rng(0)
    for name, p0_base, bounds, func, k, sym_builder in library:
        best_popt, best_mae = None, float("inf")
        for m in range(6):
            p0 = list(p0_base)
            if m:
                j = rng.normal(0, 0.3, len(p0))
                p0 = [float(np.clip(v+dj, lo, hi))
                      for v, dj, lo, hi in zip(p0, j, bounds[0], bounds[1])]
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    popt, _ = curve_fit(func, xs, ys, p0=p0, bounds=bounds,
                                        maxfev=20000)
            except Exception:
                continue
            yp = func(xs, *popt)
            if not np.all(np.isfinite(yp)): continue
            mae = float(np.mean(np.abs(yp - ys)))
            if mae < best_mae:
                best_mae, best_popt = mae, popt
        if best_popt is None: continue
        yp = func(xs, *best_popt)
        rss = float(np.sum((yp - ys)**2))
        bic = n * np.log(rss/n + 1e-15) + k * np.log(n)
        try:
            expr = sym_builder(best_popt, x_sym)
        except Exception:
            continue
        candidates.append({
            'source': 'templ_lib',
            'expr': str(expr),
            'mae': float(np.mean(np.abs(yp - ys))),
            'bic': float(bic), 'n_params': k, 'bits': 1e9,
            'sympy_expr': expr,
        })

    if not candidates: return None
    candidates.sort(key=lambda p: p['bic'])
    return candidates[0]


def _predict(expr, data, var_names):
    if expr is None: return None
    x_syms = [sp.Symbol(v) for v in var_names]
    xs = [np.array([d[0][v] for d in data], dtype=float) for v in var_names]
    ys = np.array([d[1] for d in data], dtype=float)
    try:
        f = sp.lambdify(x_syms, expr, modules=["numpy"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with np.errstate(all="ignore"):
                yp = f(*xs)
        if np.isscalar(yp): yp = np.full(len(ys), float(yp))
        yp = np.asarray(yp, dtype=float)
        if not np.all(np.isfinite(yp)): return None
        return yp, ys
    except Exception:
        return None


def _bic_pick(results, data, var_names):
    ys = np.array([d[1] for d in data], dtype=float)
    n = len(ys)
    scored = []
    for t in results:
        raw = getattr(t, "raw", {}) or {}
        e = raw.get("sympy_expr")
        if e is None: continue
        pred = _predict(e, data, var_names)
        if pred is None: continue
        yp, _ = pred
        rss = float(np.sum((yp - ys)**2))
        k = getattr(t, "n_params", 4)
        bic = n * np.log(rss/n + 1e-15) + k * np.log(n)
        mae_raw = getattr(t, "mae", None)
        if mae_raw is None: mae_raw = getattr(t, "mre", float("inf"))
        scored.append({
            "source": "templ", "expr": t.expr_str,
            "mae": float(mae_raw), "bic": float(bic), "n_params": k,
            "bits": float(getattr(t, "mdl_bits", 1e9)),
            "sympy_expr": e,
        })
    if not scored: return None
    scored.sort(key=lambda p: p["bic"])
    return scored[0]


def _fit_single_2d(data, var_names):
    from purediscover_templates import search_templates_2d
    return _bic_pick(search_templates_2d(data, var_names, max_mre=0.50,
                                          n_multistart=10, seed=0),
                     data, var_names)


def _fit_single_nd(data, var_names):
    from purediscover_templates import search_templates_nd
    return _bic_pick(search_templates_nd(data, var_names, max_mre=0.50,
                                          n_multistart=10, seed=0),
                     data, var_names)


def _fit_single_template(data, var_names):
    n = len(var_names)
    if n == 1: return _fit_single_1d(data, var_names)
    if n == 2: return _fit_single_2d(data, var_names)
    return _fit_single_nd(data, var_names)


def discover_recursive(data, var_names, *, max_depth=3,
                       noise_floor_factor=0.02, residual_var_factor=1.5,
                       time_budget=25.0, verbose=False,
                       bic_improvement_threshold=10.0):
    ys = np.array([d[1] for d in data], dtype=float)
    noise_floor = noise_floor_factor * float(np.max(np.abs(ys)) + 1e-12)

    base = _fit_single_template(data, var_names)
    if base is None:
        return {"laws": [], "composed_expr": "?", "composition_mode": "none",
                "final_mae": float("inf"), "depth": 0}

    laws = [base]
    base_expr = base["sympy_expr"]
    base_mae = base["mae"]
    base_bic = base["bic"]
    composition_mode = "single"

    if verbose:
        print(f"[depth 0] {base['source']}: mae={base_mae:.3e} bic={base_bic:.1f}")
        print(f"          {base['expr'][:100]}")

    current_data = data
    x_syms = [sp.Symbol(v) for v in var_names]

    for depth in range(1, max_depth):
        pred = _predict(base_expr, current_data, var_names)
        if pred is None: break
        yp, ys_cur = pred
        residuals = ys_cur - yp
        var_res = float(np.var(residuals))
        if var_res < (residual_var_factor * noise_floor) ** 2:
            if verbose: print(f"[depth {depth}] residuals flat -> stop")
            break

        res_data = [({v: current_data[i][0][v] for v in var_names},
                     float(residuals[i])) for i in range(len(residuals))]
        L2_add = _fit_single_template(res_data, var_names)

        L2_mul = None
        ratio_data = None
        try:
            xs = [np.array([d[0][v] for v in current_data], dtype=float)
                  for v in var_names]
            f = sp.lambdify(x_syms, base_expr, modules=["numpy"])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with np.errstate(all="ignore"):
                    yp_div = f(*xs)
            if np.isscalar(yp_div): yp_div = np.full(len(ys_cur), float(yp_div))
            yp_div = np.asarray(yp_div, dtype=float)
            mask = np.abs(yp_div) > 1e-3
            if mask.sum() >= max(5, len(ys_cur)//2):
                ratio_data = [({v: current_data[i][0][v] for v in var_names},
                               float(ys_cur[i]/yp_div[i]))
                              for i in np.where(mask)[0]]
                L2_mul = _fit_single_template(ratio_data, var_names)
        except Exception:
            pass

        best = None
        n_total = len(ys)
        for mode, L2, src_data in [("additive", L2_add, res_data),
                                    ("multiplicative", L2_mul, ratio_data)]:
            if L2 is None: continue
            try:
                composed = ((base_expr + L2["sympy_expr"]) if mode == "additive"
                            else (base_expr * L2["sympy_expr"]))
                pred_c = _predict(composed, data, var_names)
                if pred_c is None: continue
                yp_c, _ = pred_c
                rss = float(np.sum((yp_c - ys)**2))
                k_tot = base["n_params"] + L2["n_params"]
                bic_c = n_total * np.log(rss/n_total + 1e-15) + k_tot * np.log(n_total)
                mae_c = float(np.mean(np.abs(yp_c - ys)))
                cand = {"mode": mode, "L2": L2, "composed": composed,
                        "bic": bic_c, "mae": mae_c, "data": src_data}
                if best is None or bic_c < best["bic"]:
                    best = cand
            except Exception:
                continue

        if best is None: break
        if best["bic"] < base_bic - bic_improvement_threshold:
            base_expr = best["composed"]
            base_mae = best["mae"]
            base_bic = best["bic"]
            composition_mode = best["mode"]
            current_data = best["data"]
            laws.append(best["L2"])
            if verbose:
                print(f"[depth {depth}] {best['mode']}: "
                      f"mae={base_mae:.3e} bic={base_bic:.1f}")
                print(f"          {str(base_expr)[:120]}")
        else:
            if verbose: print(f"[depth {depth}] composition rejected")
            break

    return {
        "laws": laws,
        "composed_expr": str(base_expr),
        "composition_mode": composition_mode,
        "final_mae": base_mae,
        "depth": len(laws),
    }
