"""Dual-representation search: find coordinate transforms that lower MDL."""
from __future__ import annotations

import contextlib
import io
import math
from dataclasses import dataclass
from itertools import combinations, product

import numpy as np

from purediscover.discover import discover_and_classify
from purediscover.mdl import mdl_score


TRANSFORMS = {
    "id":     lambda x: x,
    "log":    lambda x: np.log(x) if np.all(x > 0) else None,
    "log1p":  lambda x: np.log1p(x) if np.all(x > -1) else None,
    "sqrt":   lambda x: np.sqrt(x) if np.all(x >= 0) else None,
    "square": lambda x: x ** 2,
    "cube":   lambda x: x ** 3,
    "inv":    lambda x: 1.0 / x if np.all(np.abs(x) > 1e-12) else None,
    "exp":    lambda x: np.exp(np.clip(x, -50, 50)),
}


@dataclass
class DualCandidate:
    transforms: dict
    dual_var_names: list
    law: object
    law_str: str
    mdl: float
    improvement: float
    candidate_dict: dict


def _extract_expr(c):
    for k in ("expression", "expr", "formula"):
        if k in c:
            return c[k]
    return None


def _extract_mdl(expr, data, var_names):
    try:
        s = mdl_score(expr, data, residual_tol=0.05)
    except Exception:
        return float("inf")
    if not isinstance(s, dict):
        return float("inf")
    for k in ("mdl", "score", "bits", "total"):
        if k in s:
            try:
                return float(s[k])
            except (TypeError, ValueError):
                pass
    return float("inf")


def _best_mdl(candidates, data, var_names):
    best_mdl, best_c = float("inf"), None
    for c in candidates:
        e = _extract_expr(c)
        if e is None:
            continue
        m = _extract_mdl(e, data, var_names)
        if m < best_mdl:
            best_mdl, best_c = m, c
    return best_mdl, best_c


def _subset_vars_used(law_str, var_names):
    return [v for v in var_names if v in law_str]


def _build_transformed_data(data, var_names, transform_map):
    cols = {v: np.array([row[0][v] for row in data], dtype=float)
            for v in var_names}
    ys = np.array([row[1] for row in data], dtype=float)
    new_cols = {}
    new_names = []
    for v in var_names:
        tname = transform_map.get(v, "id")
        f = TRANSFORMS[tname]
        try:
            tv = f(cols[v])
        except Exception:
            return None, []
        if tv is None:
            return None, []
        nv = v if tname == "id" else f"{tname}_{v}"
        new_cols[nv] = tv
        new_names.append(nv)
    new_data = [{nv: float(new_cols[nv][i]) for nv in new_names}
                for i in range(len(ys))]
    return list(zip(new_data, ys.tolist())), new_names


def _small_budget_run(data, var_names, t_direct=10.0, t_compose=20.0,
                      tol=0.10, quiet=True):
    try:
        if quiet:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                return discover_and_classify(
                    data, var_names,
                    time_budget_direct=t_direct,
                    time_budget_compose=t_compose,
                    tol=tol,
                )
        return discover_and_classify(
            data, var_names,
            time_budget_direct=t_direct,
            time_budget_compose=t_compose,
            tol=tol,
        )
    except Exception:
        return []


def search_dual(data, var_names, *, baseline_candidates=None,
                min_improvement=1.30, max_combo_size=2,
                time_budget_direct=10.0, time_budget_compose=20.0,
                tol=0.10, quiet=True, progress=True):
    if baseline_candidates is None:
        baseline_candidates = _small_budget_run(
            data, var_names, time_budget_direct, time_budget_compose, tol,
            quiet=quiet)
    baseline_mdl, baseline_c = _best_mdl(baseline_candidates, data, var_names)
    baseline_law_str = str(_extract_expr(baseline_c)) if baseline_c else "?"
    if baseline_mdl == float("inf"):
        return {"baseline_mdl": float("inf"), "baseline_law": baseline_law_str,
                "dual": None, "attempted": 0, "valid": 0,
                "note": "no valid baseline"}
    used = _subset_vars_used(baseline_law_str, var_names) or var_names
    if len(used) > 4:
        used = used[:4]
    tnames = list(TRANSFORMS.keys())
    best_dual, best_mdl = None, baseline_mdl
    attempted = valid = 0
    for combo_size in range(1, max_combo_size + 1):
        for vars_subset in combinations(used, combo_size):
            for tcombo in product(tnames, repeat=combo_size):
                if all(t == "id" for t in tcombo):
                    continue
                tm = {v: "id" for v in var_names}
                for v, t in zip(vars_subset, tcombo):
                    tm[v] = t
                attempted += 1
                if progress and attempted % 20 == 0:
                    print(f"    ... tried {attempted}, valid {valid}, "
                          f"best_mdl={best_mdl:.1f}")
                nd, nn = _build_transformed_data(data, var_names, tm)
                if nd is None:
                    continue
                valid += 1
                cands = _small_budget_run(nd, nn, time_budget_direct,
                                          time_budget_compose, tol, quiet=quiet)
                m, c = _best_mdl(cands, nd, nn)
                if c is None or m == float("inf"):
                    continue
                if m < best_mdl / min_improvement:
                    best_mdl = m
                    e = _extract_expr(c)
                    best_dual = DualCandidate(
                        transforms={v: tm[v] for v in vars_subset},
                        dual_var_names=nn, law=e, law_str=str(e),
                        mdl=m, improvement=baseline_mdl / m,
                        candidate_dict=c)
    return {"baseline_mdl": baseline_mdl, "baseline_law": baseline_law_str,
            "dual": best_dual, "attempted": attempted, "valid": valid}
