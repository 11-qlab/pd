"""Real physics templates for implicit/explicit law discovery.

Each template dict has:
  fn:           (*inputs, observable, params) -> float
  n_params:     number of free params
  default:      starting guess
  variables:    names of input variables
  observable:   name of the solved-for variable
  bracket:      (lo, hi) for solving the observable
  kind:         "implicit" (needs solver) | "explicit" (direct)
  domain:       classification

The library is curated. If the true law is not here, the pipeline fits the
closest template and gets it wrong silently. Pair with a "none fit" check.
"""
from __future__ import annotations
import numpy as np


# ---------------------------------------------------------------- fluid / pipe
def _colebrook(Re, ks_D, f, p):
    a, b, c, d, e = p
    inner = np.maximum(ks_D / c + d / (Re * np.sqrt(np.maximum(f, 1e-30))), 1e-30)
    return a / np.sqrt(np.maximum(f, 1e-30)) + b * np.log10(inner) + e


def _hazen_williams(L, Q, hf, p):
    C, k = p
    D = L  # length as pipe-ID proxy; caller supplies (L, Q) -> hf
    return hf - k * (Q / C) ** 1.852 * D


def _manning(R, S, v, p):
    n = p[0]
    return v - (1.0 / n) * R ** (2/3) * np.sqrt(S)


# ---------------------------------------------------------------- thermo
def _antoine(T, P, p):
    A, B, C = p
    return np.log10(np.maximum(P, 1e-30)) - (A - B / (C + T))


def _arrhenius(T, k, p):
    A, Ea = p
    R = 8.314
    return k - A * np.exp(-Ea / (R * T))


def _van_der_waals(P, V, p):
    a, b, T, R = p
    return (P + a / V**2) * (V - b) - R * T


def _redlich_kwong(P, V, p):
    a, b, T, R = p
    return P - R * T / (V - b) + a / (np.sqrt(T) * V * (V + b))


# ---------------------------------------------------------------- surface
def _langmuir(c, q, p):
    q_max, K = p
    return q - q_max * K * c / (1 + K * c)


def _freundlich(c, q, p):
    K, n = p
    return q - K * c ** (1.0 / n)


def _bet(P, v, p):
    vm, C, P0 = p
    return v - vm * C * P / ((P0 - P) * (1 + (C - 1) * P / P0))


# ---------------------------------------------------------------- enzyme
def _michaelis_menten(s, v, p):
    vmax, Km = p
    return v - vmax * s / (Km + s)


def _hill(L, y, p):
    Kd, n = p
    return y - L**n / (Kd**n + L**n)


# ---------------------------------------------------------------- electrochem
def _butler_volmer(eta, j, p):
    j0, alpha, T = p
    F, R = 96485.0, 8.314
    return j - j0 * (np.exp(alpha * F * eta / (R * T)) -
                     np.exp(-(1 - alpha) * F * eta / (R * T)))


def _nernst(Q, E, p):
    E0, n = p
    F, R, T = 96485.0, 8.314, 298.15
    return E - (E0 - (R * T / (n * F)) * np.log(np.maximum(Q, 1e-30)))


TEMPLATES = {
    # pipe
    "colebrook": dict(fn=_colebrook, n_params=5,
                      default=[1.0, 2.0, 3.7, 2.51, 0.0],
                      variables=["Re", "ks_D"], observable="f",
                      bracket=(1e-4, 0.4), kind="implicit",
                      domain="fluid dynamics"),
    "hazen_williams": dict(fn=_hazen_williams, n_params=2,
                           default=[100.0, 10.67],
                           variables=["L", "Q"], observable="hf",
                           bracket=(1e-6, 1e4), kind="explicit",
                           domain="fluid dynamics"),
    "manning": dict(fn=_manning, n_params=1, default=[0.013],
                    variables=["R", "S"], observable="v",
                    bracket=(1e-4, 100.0), kind="explicit",
                    domain="fluid dynamics"),
    # thermo
    "antoine": dict(fn=_antoine, n_params=3, default=[5.0, 1500.0, 200.0],
                    variables=["T"], observable="P",
                    bracket=(1e-3, 1e7), kind="explicit",
                    domain="vapor pressure"),
    "arrhenius": dict(fn=_arrhenius, n_params=2, default=[1e10, 5e4],
                      variables=["T"], observable="k",
                      bracket=(1e-30, 1e30), kind="explicit",
                      domain="reaction kinetics"),
    "van_der_waals": dict(fn=_van_der_waals, n_params=4,
                          default=[0.364, 4.27e-5, 300.0, 8.314],
                          variables=["P"], observable="V",
                          bracket=(1e-5, 1.0), kind="implicit",
                          domain="equation of state"),
    "redlich_kwong": dict(fn=_redlich_kwong, n_params=4,
                          default=[1.0, 3e-5, 300.0, 8.314],
                          variables=["P"], observable="V",
                          bracket=(1e-5, 1.0), kind="implicit",
                          domain="equation of state"),
    # surface
    "langmuir": dict(fn=_langmuir, n_params=2, default=[1.0, 1.0],
                     variables=["c"], observable="q",
                     bracket=(0.0, 1e6), kind="explicit",
                     domain="adsorption"),
    "freundlich": dict(fn=_freundlich, n_params=2, default=[1.0, 2.0],
                       variables=["c"], observable="q",
                       bracket=(0.0, 1e6), kind="explicit",
                       domain="adsorption"),
    "bet": dict(fn=_bet, n_params=3, default=[1.0, 50.0, 1.0],
                variables=["P"], observable="v",
                bracket=(1e-6, 1e6), kind="explicit",
                domain="adsorption"),
    # enzyme
    "michaelis_menten": dict(fn=_michaelis_menten, n_params=2,
                             default=[1.0, 1.0],
                             variables=["s"], observable="v",
                             bracket=(0.0, 1e6), kind="explicit",
                             domain="enzyme kinetics"),
    "hill": dict(fn=_hill, n_params=2, default=[1.0, 2.0],
                 variables=["L"], observable="y",
                 bracket=(0.0, 1.0), kind="explicit",
                 domain="cooperativity"),
    # electrochem
    "butler_volmer": dict(fn=_butler_volmer, n_params=3,
                          default=[1.0, 0.5, 298.15],
                          variables=["eta"], observable="j",
                          bracket=(-1e6, 1e6), kind="explicit",
                          domain="electrochemistry"),
    "nernst": dict(fn=_nernst, n_params=2, default=[1.0, 1.0],
                   variables=["Q"], observable="E",
                   bracket=(-1e6, 1e6), kind="explicit",
                   domain="electrochemistry"),
}


def list_templates(domain=None):
    if domain is None:
        return sorted(TEMPLATES.keys())
    return sorted(k for k, v in TEMPLATES.items() if v["domain"] == domain)


def evaluate_template(template_name, inputs_dict, observable_value, params):
    """Compute F. inputs_dict has all variable names except observable."""
    spec = TEMPLATES[template_name]
    args = [inputs_dict[v] for v in spec["variables"]] + [observable_value]
    return spec["fn"](*args, params)


def check_all_templates_fail(data, rss_floor_frac=0.1):
    """Return list of templates whose best fit has rss > rss_floor_frac * var(y).

    None fitting is the 'novel physics' signal. If this returns [] (all fit),
    or ['all'], the data is either trivial or genuinely outside the library.
    """
    from scipy.optimize import least_squares
    y_vals = np.array([d["y"] for d in data])
    var_y = float(np.var(y_vals))
    if var_y <= 0:
        return sorted(TEMPLATES.keys())
    out = []
    for name in TEMPLATES:
        spec = TEMPLATES[name]
        def res(p):
            rs = []
            for d in data:
                inputs = {v: d["x"][i] for i, v in enumerate(spec["variables"])}
                rs.append(evaluate_template(name, inputs, d["y"], p))
            return np.array(rs)
        try:
            sol = least_squares(res, np.array(spec["default"], dtype=float),
                                max_nfev=2000)
        except Exception:
            out.append(name)
            continue
        if sol.fun @ sol.fun > rss_floor_frac * var_y * len(data):
            out.append(name)
    return out


def _is_degenerate(template_name, params, data):
    spec = TEMPLATES[template_name]
    lo, hi = spec["bracket"]
    d0 = data[0]
    inputs = {v: d0["x"][i] for i, v in enumerate(spec["variables"])}
    try:
        F_lo = evaluate_template(template_name, inputs, lo, params)
        F_hi = evaluate_template(template_name, inputs, hi, params)
    except Exception:
        return True
    if not (np.isfinite(F_lo) and np.isfinite(F_hi)):
        return True
    if abs(F_lo) < 1e-4 or abs(F_hi) < 1e-4:
        return True
    return F_lo * F_hi >= 0


def fit_template(template_name, data, init_params=None, n_starts=5):
    from scipy.optimize import least_squares
    spec = TEMPLATES[template_name]
    p_default = np.array(spec["default"], dtype=float)
    def res(params):
        out = []
        for d in data:
            inputs = {v: d["x"][i] for i, v in enumerate(spec["variables"])}
            try:
                v = evaluate_template(template_name, inputs, d["y"], params)
            except Exception:
                v = np.nan
            out.append(v)
        return np.array(out)
    candidates = [p_default]
    if init_params is not None:
        candidates.append(np.array(init_params, dtype=float))
    rng = np.random.default_rng(0)
    for _ in range(n_starts):
        scale = rng.uniform(0.5, 2.0, len(p_default))
        candidates.append(p_default * scale)
    best = None
    for c0 in candidates:
        try:
            sol = least_squares(res, c0, max_nfev=3000)
        except Exception:
            continue
        if _is_degenerate(template_name, sol.x, data):
            continue
        rss = float(np.sum(sol.fun ** 2))
        if best is None or rss < best[1]:
            best = (sol.x, rss)
    if best is None:
        sol = least_squares(res, p_default, max_nfev=3000)
        rss = float(np.sum(sol.fun ** 2))
        return sol.x, float(np.sqrt(rss / max(len(data), 1)))
    rss = best[1]
    return best[0], float(np.sqrt(rss / max(len(data), 1)))
