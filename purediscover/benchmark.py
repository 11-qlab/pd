"""Rediscovery benchmark. Routes everything to brute_search.

Budget sizing:
    difficulty <= 5 : max_nodes=12, time_budget=120s
    difficulty >= 6 : max_nodes=16, time_budget=600s

NOTE on sqrt/sin/tan/abs/inv/neg: brute_search's dispatcher has no
emission path for these ops (verified by examples/probe_ops.py — 0
candidates in <1s). More time does not help. Use
examples/benchmark_peeled.py for a peel-transform wrapper that
recovers sqrt-shaped laws via data transformation.
"""
from __future__ import annotations
import random
import time
import sympy as sp

from purediscover.symbolic import Grammar
from purediscover.rich_search import brute_search
from purediscover.resource.laws import Law, REGISTRY


def _build_data(law: Law, n: int = 40, seed: int = 0,
                noise: float = 0.0) -> list[tuple[dict[str, float], float]]:
    rng = random.Random(seed)
    vars_ = list(law.variables.keys())
    data = []
    for _ in range(n):
        env = {v: rng.uniform(0.5, 5.0) for v in vars_}
        y = float(law.expr.subs({sp.Symbol(k): val for k, val in env.items()}))
        if noise > 0:
            y += rng.gauss(0, noise * max(1e-9, abs(y)))
        data.append((env, y))
    return data


def _grammar_for(law: Law) -> Grammar:
    ops = list(law.ops_needed) or ["add"]
    norm: list[str] = []
    for op in ops:
        if op == "pow3":
            norm.append("pow")
        elif op == "sqrt":
            norm.extend(["sqrt", "pow"])
        elif op == "pow":
            norm.append("pow")
        else:
            norm.append(op)
    seen: set[str] = set()
    norm = [o for o in norm if not (o in seen or seen.add(o))]
    return Grammar(variables=list(law.variables.keys()),
                   constants=[1.0], max_depth=4, ops=norm)


def _equivalent_up_to_scalar(a: sp.Expr, b: sp.Expr, atol: float = 0.02) -> bool:
    try:
        if sp.simplify(a - b) == 0:
            return True
    except Exception:
        pass
    try:
        ratio = sp.simplify(a / b)
        if not ratio.free_symbols:
            return True
    except Exception:
        pass
    try:
        syms = list(a.free_symbols | b.free_symbols)
        if not syms:
            return False
        rng = random.Random(42)
        ratios = []
        for _ in range(20):
            sub = {s: rng.uniform(0.6, 3.0) for s in syms}
            va = float(a.subs(sub)); vb = float(b.subs(sub))
            if abs(vb) < 1e-9:
                continue
            ratios.append(va / vb)
        if not ratios:
            return False
        r0 = ratios[0]
        return all(abs(r - r0) <= atol * max(1.0, abs(r0)) for r in ratios)
    except Exception:
        return False


def run_one(law: Law, time_budget: float | None = None,
            max_nodes: int | None = None, tol: float = 0.02) -> dict:
    if time_budget is None:
        time_budget = 600.0 if law.difficulty >= 6 else 120.0
    if max_nodes is None:
        max_nodes = 16 if law.difficulty >= 6 else 12

    data = _build_data(law, n=40, seed=0, noise=0.0)
    g = _grammar_for(law)
    t0 = time.time()
    try:
        cands = brute_search(g, data, max_nodes=max_nodes,
                             max_per_level=50000, tol=tol,
                             time_budget=time_budget)
    except Exception as e:
        return {"name": law.name, "found": False, "reason": str(e),
                "elapsed": time.time() - t0}
    elapsed = time.time() - t0
    if not cands:
        return {"name": law.name, "found": False, "reason": "no candidates",
                "elapsed": elapsed}
    best = cands[0]["expr"]
    matched = _equivalent_up_to_scalar(best, law.expr)
    return {"name": law.name, "found": matched,
            "found_expr": str(best), "target_expr": str(law.expr),
            "bits": cands[0]["score"]["bits"], "elapsed": elapsed}


def run_all(laws: list[Law] | None = None, time_budget: float | None = None,
            max_difficulty: int = 10) -> dict:
    laws = laws or [l for l in REGISTRY if l.difficulty <= max_difficulty]
    results = []
    n_ok = 0
    for law in laws:
        r = run_one(law, time_budget=time_budget)
        r["domain"] = law.domain
        r["difficulty"] = law.difficulty
        results.append(r)
        n_ok += int(r["found"])
        mark = "✓" if r["found"] else "✗"
        print(f"  {mark}  {law.domain:16s} {law.name:22s} "
              f"n={law.difficulty:<2}  {r['elapsed']:5.1f}s  "
              f"{r.get('found_expr', r.get('reason', ''))[:50]}")
    summary = {
        "total": len(results),
        "found": n_ok,
        "rate": n_ok / max(1, len(results)),
        "by_domain": {},
        "results": results,
    }
    for r in results:
        d = summary["by_domain"].setdefault(r["domain"], {"n": 0, "found": 0})
        d["n"] += 1
        d["found"] += int(r["found"])
    return summary
