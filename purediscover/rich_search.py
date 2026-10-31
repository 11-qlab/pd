"""Rich search + brute-force BFS by node count.

Fixes:
  1. levels[n] stores expressions only (was storing scored tuples, which
     crashed _canonical when a constant-only expression carried an empty
     variable set).
  2. Guard rail raised to 10M; max_per_level default lowered to 20000.
     This skips L10 blow-up pairs (5,5) and (4,6) that dominate runtime
     without contributing to the add+sub+pow path, and lets (4,5) finish.
  3. np.seterr(all='ignore') at module load to silence overflow warnings
     from deep power chains; results are still rejected via isfinite.
  4. _canonical is robust to sets, frozensets, and empty tuples.
"""
from __future__ import annotations
import math
import time
import sympy as sp
import numpy as np

np.seterr(all="ignore")

from purediscover.mdl import mdl_score, _scale
from purediscover.symbolic import Grammar, _apply, _leaves


def _canonical(expr):
    if isinstance(expr, (int, float, str)):
        return expr
    if isinstance(expr, (set, frozenset)):
        return tuple(sorted((_canonical(e) for e in expr), key=str))
    if not isinstance(expr, (tuple, list)):
        return expr
    if not expr:
        return ()
    op, *args = expr
    cargs = [_canonical(a) for a in args]
    if op in ("add", "mul"):
        cargs = tuple(sorted(cargs, key=str))
    else:
        cargs = tuple(cargs)
    return (op, *cargs)


def _node_count(expr):
    if isinstance(expr, (int, float, str)):
        return 1
    if not isinstance(expr, (tuple, list)) or not expr:
        return 1
    return 1 + sum(_node_count(a) for a in expr[1:])


def _vars_used(expr, out):
    if isinstance(expr, str):
        out.add(expr); return
    if not isinstance(expr, (tuple, list)) or not expr:
        return
    for a in expr[1:]:
        _vars_used(a, out)


def _eval_np(expr, env):
    if isinstance(expr, str):
        return env[expr]
    if isinstance(expr, (int, float)):
        return np.full_like(next(iter(env.values())), float(expr), dtype=float)
    if not isinstance(expr, (tuple, list)) or not expr:
        return np.full_like(next(iter(env.values())), np.nan, dtype=float)
    op = expr[0]
    a = _eval_np(expr[1], env)
    if len(expr) == 2:
        if op == "pow2": return a * a
        if op == "pow3": return a * a * a
        if op == "log":
            return np.where(a > 0, np.log(np.where(a > 0, a, 1.0)), np.inf)
        if op == "exp":
            return np.where(a < 700, np.exp(np.where(a < 700, a, 0.0)), np.inf)
        return np.full_like(a, np.inf)
    b = _eval_np(expr[2], env)
    if op == "add": return a + b
    if op == "sub": return a - b
    if op == "mul": return a * b
    if op == "div":
        return np.where(b != 0, a / b, np.inf)
    return np.full_like(a, np.inf)


def _to_sympy(expr):
    if isinstance(expr, str): return sp.Symbol(expr)
    if isinstance(expr, (int, float)): return sp.Float(float(expr))
    if not isinstance(expr, (tuple, list)) or not expr:
        return None
    op = expr[0]
    args = [_to_sympy(a) for a in expr[1:]]
    if any(a is None for a in args): return None
    try:
        if op == "add": return args[0] + args[1]
        if op == "sub": return args[0] - args[1]
        if op == "mul": return args[0] * args[1]
        if op == "div": return args[0] / args[1]
        if op == "pow2": return args[0] ** 2
        if op == "pow3": return args[0] ** 3
        if op == "log": return sp.log(args[0])
        if op == "exp": return sp.exp(args[0])
    except Exception:
        return None
    return None


def _fit_check(p, y_arr, scale):
    if p is None or not np.all(np.isfinite(p)):
        return float("inf"), float("inf"), 1.0
    r_raw = float(np.max(np.abs(p - y_arr)) / scale)
    denom = float(np.sum(p * p))
    if denom < 1e-12:
        return r_raw, float("inf"), 1.0
    c = float(np.sum(p * y_arr)) / denom
    if not math.isfinite(c) or c == 0:
        return r_raw, float("inf"), 1.0
    r_scaled = float(np.max(np.abs(c * p - y_arr)) / scale)
    return r_raw, r_scaled, c


def _prune(scored, K, var_list):
    if len(scored) <= K:
        return scored
    small = [t for t in scored if t[3] <= 5]
    if len(small) >= K:
        return sorted(small, key=lambda t: (t[1], t[3]))[:K]
    remaining = K - len(small)
    rest = [t for t in scored if t[3] > 5]
    by_fit = sorted(rest, key=lambda t: t[1])[:remaining // 3]
    by_size = sorted(rest, key=lambda t: (t[3], t[1]))[:remaining // 3]
    both_vars = [t for t in rest if len(t[4]) >= len(var_list)]
    by_div = sorted(both_vars, key=lambda t: (t[3], t[1]))[:max(0, remaining - len(by_fit) - len(by_size))]
    seen, out = set(), []
    for t in small + by_fit + by_size + by_div:
        if t[0] not in seen:
            seen.add(t[0])
            out.append(t)
    return out[:K]


def brute_search(
    grammar: Grammar,
    data: list[tuple[dict[str, float], float]],
    max_nodes: int = 12,
    max_per_level: int = 20000,
    tol: float = 0.02,
    time_budget: float = 600.0,
    guard_pairs: int = 10_000_000,
) -> list[dict]:
    t0 = time.time()
    y_arr = np.array([y for _, y in data], dtype=float)
    scale = _scale(list(y_arr))
    env_keys = list(data[0][0].keys())
    env = {k: np.array([row[0][k] for row in data], dtype=float) for k in env_keys}
    var_list = list(grammar.variables)

    def score(e):
        try:
            p = _eval_np(e, env)
        except Exception:
            return float("inf"), float("inf"), set()
        r_raw, r_scaled, _ = _fit_check(p, y_arr, scale)
        vars_used: set = set()
        _vars_used(e, vars_used)
        return r_raw, r_scaled, vars_used

    leaves: list = list(grammar.variables) + list(grammar.constants)
    levels: dict[int, list] = {1: [_canonical(l) for l in leaves]}
    ops = list(grammar.ops)
    fits: list[dict] = []

    for n in range(2, max_nodes + 1):
        if time.time() - t0 > time_budget:
            print(f"  [time budget hit at n={n}]", flush=True)
            break

        cand_set: set = set()

        for unary, needs in (("pow2", "pow"), ("pow3", "pow"), ("log", "log"), ("exp", "exp")):
            if needs not in ops: continue
            for child in levels.get(n - 1, ()):
                cand_set.add(_canonical((unary, child)))

        for op in ops:
            if op in ("log", "exp", "pow"): continue
            for i in range(1, n - 1):
                j = n - 1 - i
                if j < 1: continue
                Li = levels.get(i, ())
                Lj = levels.get(j, ())
                if not Li or not Lj: continue
                if len(Li) * len(Lj) > guard_pairs:
                    continue
                for a in Li:
                    for b in Lj:
                        cand_set.add(_canonical((op, a, b)))

        cand = list(cand_set)
        if not cand:
            continue
        print(f"  L{n}: {len(cand)} candidates", flush=True)

        scored: list = []
        for idx, e in enumerate(cand):
            if (idx & 4095) == 0 and time.time() - t0 > time_budget:
                break
            r_raw, r_scaled, vars_used = score(e)
            if math.isfinite(r_scaled):
                scored.append((e, r_scaled, r_raw, _node_count(e), vars_used))

        for e, r_scaled, r_raw, nc, vars_used in scored:
            if r_scaled <= tol * 1.5:
                sp_expr = _to_sympy(e)
                if sp_expr is None: continue
                s = mdl_score(sp_expr, data, residual_tol=tol)
                if s["ok"]:
                    fits.append({"expr": s.get("expr", sp_expr), "score": s, "nodes": n})

        if fits:
            fits.sort(key=lambda r: r["score"]["bits"])
            print(f"  -> {len(fits)} fits found at n={n}", flush=True)
            return fits[:10]

        pruned = _prune(scored, max_per_level, var_list)
        levels[n] = [t[0] for t in pruned]

    return fits[:10]


def discover_law_rich(grammar, data, n_candidates=6000, tol=0.02,
                      beam=64, size_beam=32, time_budget=60.0):
    return brute_search(
        grammar, data,
        max_nodes=max(6, grammar.max_depth + 4),
        max_per_level=max(20000, n_candidates),
        tol=tol,
        time_budget=time_budget,
    )
