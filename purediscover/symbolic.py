"""Expression-tree search with MDL gate.

Beam search over grammar-generated expressions.
Each depth: expand every beam node with every op, simplify, dedup by
srepr, score, keep top-beam. Deterministic given seed and time budget.
"""
from __future__ import annotations
import random
import time
from dataclasses import dataclass, field
import sympy as sp

from purediscover.mdl import mdl_score


@dataclass
class Grammar:
    variables: list[str]
    constants: list[float] = field(default_factory=lambda: [1.0])
    max_depth: int = 3
    ops: list[str] = field(default_factory=lambda: ["add", "mul", "div", "pow"])


def _leaves(g: Grammar):
    for v in g.variables:
        yield sp.Symbol(v)
    for c in g.constants:
        yield sp.Float(c)


def _apply(op: str, a: sp.Expr, b: sp.Expr):
    if op == "add":
        return a + b
    if op == "mul":
        return a * b
    if op == "div":
        if b.is_zero:
            return None
        return a / b
    return None


def _depth0(g: Grammar):
    leaves = list(_leaves(g))
    out = list(leaves)
    for op in g.ops:
        if op == "pow":
            for a in leaves:
                for k in (2, 3):
                    out.append(a ** k)
        else:
            for a in leaves:
                for b in leaves:
                    e = _apply(op, a, b)
                    if e is not None:
                        out.append(e)
    return out


def _expand(exprs: list[sp.Expr], g: Grammar):
    leaves = list(_leaves(g))
    out = []
    for e in exprs:
        for op in g.ops:
            if op == "pow":
                for k in (2, 3):
                    out.append(e ** k)
                continue
            for l in leaves:
                x = _apply(op, e, l)
                if x is not None:
                    out.append(x)
                y = _apply(op, l, e)
                if y is not None:
                    out.append(y)
    return out


def discover_law(
    grammar: Grammar,
    data: list[tuple[dict[str, float], float]],
    n_candidates: int = 5000,
    seed: int = 0,
    tol: float = 0.02,
    beam: int = 64,
    time_budget: float = 60.0,
) -> list[dict]:
    """Return top candidates sorted by MDL bits (lower is better).

    n_candidates is a soft cap on total scored expressions.
    time_budget is a hard wall-clock cap in seconds.
    """
    rng = random.Random(seed)  # noqa: F841  (reserved for future stochastic ops)
    t0 = time.time()
    seen: dict[str, dict | None] = {}
    n_scored = 0

    def consider(expr: sp.Expr):
        nonlocal n_scored
        if n_scored >= n_candidates or time.time() - t0 > time_budget:
            return None
        try:
            simp = sp.simplify(expr)
        except Exception:
            return None
        key = sp.srepr(simp)
        if key in seen:
            return None
        n_scored += 1
        s = mdl_score(simp, data, residual_tol=tol)
        if not s["ok"]:
            seen[key] = None
            return None
        rec = {"expr": s.get("expr", simp), "score": s, "key": key}
        seen[key] = rec
        return rec

    frontier = _depth0(grammar)
    depth = 1
    while frontier and depth <= grammar.max_depth:
        if n_scored >= n_candidates or time.time() - t0 > time_budget:
            break
        scored = []
        for e in frontier:
            r = consider(e)
            if r:
                scored.append(r)
        if not scored:
            break
        scored.sort(key=lambda r: r["score"]["bits"])
        beam_exprs = [r["expr"] for r in scored[:beam]]
        frontier = _expand(beam_exprs, grammar)
        depth += 1

    final = [r for r in seen.values() if r is not None]
    final.sort(key=lambda r: r["score"]["bits"])
    return final[:10]
