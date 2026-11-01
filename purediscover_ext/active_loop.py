"""Active experiment loop (restricted closing).

Robust initial fit (finer grid + top-K multi-start NM), then warm-start
refits. Without multi-start, 2% noise + few initial points lets NM land in
a wrong local minimum that warm-start never escapes. With top-K multi-start,
initial fit converges to the correct basin and warm-start refits improve.

Restricted scope: proposes times for a fixed observable; does not choose
new observable types or physical systems.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize
from purediscover_ext.observable_selection import evolve_expectation


def propose_times(J_hat, h_hat, obs_name, n_new=5, T_max=15.0, n_grid=400):
    t_grid = np.linspace(0.01, T_max, n_grid)
    eps = 1e-5
    base = np.array([evolve_expectation(J_hat, h_hat, np.array([ti]), obs_name)[0]
                     for ti in t_grid])
    sens = np.empty((2, n_grid))
    for i, ti in enumerate(t_grid):
        Jp = evolve_expectation(J_hat + eps, h_hat, np.array([ti]), obs_name)[0]
        hp = evolve_expectation(J_hat, h_hat + eps, np.array([ti]), obs_name)[0]
        sens[0, i] = (Jp - base[i]) / eps
        sens[1, i] = (hp - base[i]) / eps

    selected = []
    for _ in range(n_new):
        best_score, best_idx = -np.inf, -1
        for i in range(n_grid):
            if i in selected:
                continue
            cols = [sens[:, j] for j in selected] + [sens[:, i]]
            M = np.column_stack(cols)
            sign, logdet = np.linalg.slogdet(M @ M.T + np.eye(2) * 1e-12)
            if sign > 0 and logdet > best_score:
                best_score, best_idx = logdet, i
        if best_idx >= 0:
            selected.append(best_idx)
    return t_grid[selected]


def _loss(J, h, t, y, obs_name):
    if J <= 0 or h <= 0:
        return 1e10
    pred = evolve_expectation(J, h, t, obs_name)
    return float(np.mean((pred - y) ** 2))


def _fit_grid(t, y, obs_name, grid_n=25, top_k=5):
    """Coarse grid -> pick top-K cells -> NM from each -> return best."""
    Js = np.linspace(0.05, 2.0, grid_n)
    hs = np.linspace(0.05, 2.0, grid_n)
    cells = []
    for J0 in Js:
        for h0 in hs:
            cells.append((_loss(J0, h0, t, y, obs_name), J0, h0))
    cells.sort(key=lambda c: c[0])

    best_loss, best_J, best_h = np.inf, 0.5, 0.5
    for _, J0, h0 in cells[:top_k]:
        res = minimize(lambda p: _loss(p[0], p[1], t, y, obs_name),
                       [J0, h0], method="Nelder-Mead",
                       options={"xatol": 1e-9, "fatol": 1e-14, "maxiter": 3000})
        if res.fun < best_loss:
            best_loss = float(res.fun)
            best_J, best_h = float(res.x[0]), float(res.x[1])
    return best_J, best_h


def _fit_warm(J_prev, h_prev, t, y, obs_name):
    res = minimize(lambda p: _loss(p[0], p[1], t, y, obs_name),
                   [J_prev, h_prev], method="Nelder-Mead",
                   options={"xatol": 1e-9, "fatol": 1e-14, "maxiter": 3000})
    return float(res.x[0]), float(res.x[1])


def active_recovery(J_true, h_true, obs_name, t_init, n_iter=3, n_new=5,
                    noise_frac=0.02, seed=0):
    rng = np.random.default_rng(seed)
    t_all = np.array(t_init, dtype=float)
    y_all = evolve_expectation(J_true, h_true, t_all, obs_name)
    y_all = y_all + noise_frac * rng.standard_normal(len(y_all))

    J_hat, h_hat = _fit_grid(t_all, y_all, obs_name)
    history = [{"iter": 0, "n": len(t_all), "J_hat": J_hat, "h_hat": h_hat,
                "j_err": abs(J_hat - J_true) / J_true,
                "h_err": abs(h_hat - h_true) / h_true}]

    for it in range(1, n_iter):
        new_t = propose_times(J_hat, h_hat, obs_name, n_new=n_new,
                              T_max=max(15.0, 3 * t_all.max()))
        y_new = evolve_expectation(J_true, h_true, new_t, obs_name)
        y_new = y_new + noise_frac * rng.standard_normal(len(y_new))
        t_all = np.concatenate([t_all, new_t])
        y_all = np.concatenate([y_all, y_new])
        J_hat, h_hat = _fit_warm(J_hat, h_hat, t_all, y_all, obs_name)
        history.append({"iter": it, "n": len(t_all), "J_hat": J_hat,
                        "h_hat": h_hat,
                        "j_err": abs(J_hat - J_true) / J_true,
                        "h_err": abs(h_hat - h_true) / h_true})
    return history
