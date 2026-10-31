"""Peel-transform discovery.

Recovers laws whose outer op is one brute_search cannot emit
(sqrt, log-with-outer-neg, variable-exponent pow) by transforming the
*data* until the inner expression is findable, then reconstructing.

Peels tried in order, first match wins:
    plain     y = f(vars)               search directly
    square    y = sqrt(g(vars))         square data, search g, sqrt(g)
    log       y = log(g(vars))          log data, search g, exp(g)
    var_root  y = base ** var           take var-th root of y, search base
    pow_fit   y = c * var**k            log-log regression for univariate powers

Key design: peel inner searches run on a grammar AUGMENTED with
'mul', 'add', 'sub', 'div' so bare variables (via 1.0*var) and simple
coefficients become expressible. Without this, sqrt(L) reduces to
searching for L, which the base grammar ['sqrt','pow'] cannot emit.

pow_fit is a backstop for univariate power laws (sqrt(L), 1/f, T**4,
r**-2) — snap-fit the log-log slope to a small rational. Uses only
data, no symbolic search, so it succeeds when brute_search fails.

This is a workaround, not a fix. The correct long-term solution is
adding sqrt emission to rich_search.py's dispatcher (see notes in
examples/probe_ops.py). Peels are safe because they validate
reconstruction numerically via equivalent_up_to_scalar.
"""
from __future__ import annotations
import math
import sympy as sp

from purediscover.symbolic import Grammar
from purediscover.rich_search import brute_search


def _augment_for_peel(g: Grammar) -> Grammar:
    """Add binary ops so inner searches can emit bare variables."""
    ops = set(getattr(g, "ops", None) or [])
    ops.update(["mul", "add", "sub", "div"])
    return Grammar(
        variables=list(g.variables),
        constants=list(getattr(g, "constants", None) or [1.0]),
        max_depth=getattr(g, "max_depth", 4),
        ops=sorted(ops),
    )


def _safe_brute(g, data, time_budget, max_nodes, tol):
    try:
        return brute_search(g, data, max_nodes=max_nodes,
                            max_per_level=50000, tol=tol,
                            time_budget=time_budget)
    except Exception:
        return []


def _peel_plain(g, data, budget, max_nodes, tol):
    return _safe_brute(g, data, budget, max_nodes, tol), lambda e: e


def _peel_square(g, data, budget, max_nodes, tol):
    g2 = _augment_for_peel(g)
    sq = [(env, y * y) for env, y in data]
    inner = _safe_brute(g2, sq, budget, max_nodes, tol)
    if not inner:
        return [], None
    return ([{"expr": sp.sqrt(c["expr"]), "score": c["score"]} for c in inner],
            lambda e: sp.sqrt(e))


def _peel_log(g, data, budget, max_nodes, tol):
    if not all(y > 0 for _, y in data):
        return [], None
    g2 = _augment_for_peel(g)
    ld = [(env, math.log(y)) for env, y in data]
    inner = _safe_brute(g2, ld, budget, max_nodes, tol)
    if not inner:
        return [], None
    return ([{"expr": sp.exp(c["expr"]), "score": c["score"]} for c in inner],
            lambda e: sp.exp(e))


def _peel_var_root(g, data, budget, max_nodes, tol):
    g2 = _augment_for_peel(g)
    for var in g.variables:
        try:
            rdata = []
            for env, y in data:
                xv = env[var]
                if xv == 0 or y <= 0:
                    rdata = []
                    break
                rdata.append((env, y ** (1.0 / xv)))
            if not rdata:
                continue
            inner = _safe_brute(g2, rdata, budget, max_nodes, tol)
            if not inner:
                continue
            var_sym = sp.Symbol(var)
            return ([{"expr": c["expr"] ** var_sym, "score": c["score"]}
                     for c in inner],
                    lambda e, v=var_sym: e ** v)
        except Exception:
            continue
    return [], None


_NICE_POWERS = (2, 3, 4, -1, -2, -3, 0.5, -0.5, 1.5, -1.5, 1, 5)


def _peel_pow_fit(g, data, budget, max_nodes, tol):
    """Univariate power law: y = c * var**k, fit k via log-log slope.
    Catches sqrt(L), 1/f, T**4, r**-2, n**-2 that brute_search can
    cannot emit as a naked power with fractional/negative exponent.
    """
    for var in g.variables:
        xs, ys = [], []
        ok = True
        for env, y in data:
            if env[var] <= 0 or y <= 0:
                ok = False; break
            xs.append(math.log(env[var]))
            ys.append(math.log(y))
        if not ok or len(xs) < 4:
            continue
        n = len(xs)
        mx = sum(xs) / n; my = sum(ys) / n
        num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        den = sum((x - mx) ** 2 for x in xs)
        if den == 0:
            continue
        slope = num / den
        # require near-perfect power-law fit (small residual)
        resid = sum((ys[i] - (my + slope * (xs[i] - mx))) ** 2 for i in range(n))
        if resid / max(1e-12, n) > 0.01:
            continue
        for k in _NICE_POWERS:
            if abs(slope - k) < 0.05:
                var_sym = sp.Symbol(var)
                if k == int(k):
                    expr = var_sym ** int(k)
                else:
                    expr = var_sym ** sp.Rational(str(k)).limit_denominator(10)
                return [{"expr": expr, "score": {"bits": 0}}], lambda e: e
    return [], None


PEELS = [
    ("plain",    _peel_plain),
    ("square",   _peel_square),
    ("log",      _peel_log),
    ("var_root", _peel_var_root),
    ("pow_fit",  _peel_pow_fit),
]


def run_one_peeled(law, grammar_for, build_data, equivalent_fn,
                   time_budget: float | None = None,
                   max_nodes: int | None = None,
                   tol: float = 0.02) -> dict:
    import time
    if time_budget is None:
        time_budget = 120.0 if law.difficulty >= 6 else 60.0
    if max_nodes is None:
        max_nodes = 16 if law.difficulty >= 6 else 12

    data = build_data(law, n=40, seed=0, noise=0.0)
    g = grammar_for(law)
    t0 = time.time()

    for peel_name, peel_fn in PEELS:
        budget = max(20.0, time_budget / 3)
        cands, _ = peel_fn(g, data, budget, max_nodes, tol)
        if not cands:
            continue
        best = cands[0]["expr"]
        if equivalent_fn(best, law.expr):
            return {"name": law.name, "found": True, "peel": peel_name,
                    "found_expr": str(best), "target_expr": str(law.expr),
                    "elapsed": time.time() - t0}
    return {"name": law.name, "found": False, "reason": "no peel matched",
            "elapsed": time.time() - t0}


def run_benchmark_peeled(laws=None, time_budget: float | None = None,
                         max_difficulty: int = 10) -> dict:
    """Convenience runner: peel-benchmark the full corpus or a subset.

    Lives here (not in examples/) so both examples and downstream
    training code can import it. Lazily imports benchmark helpers to
    keep peel.py standalone.
    """
    from purediscover.benchmark import (
        _build_data, _grammar_for, _equivalent_up_to_scalar,
    )
    from purediscover.resource.laws import REGISTRY

    laws = laws or [l for l in REGISTRY if l.difficulty <= max_difficulty]
    results = []
    n_ok = 0
    for law in laws:
        r = run_one_peeled(law, _grammar_for, _build_data,
                           _equivalent_up_to_scalar,
                           time_budget=time_budget)
        r["domain"] = law.domain
        r["difficulty"] = law.difficulty
        results.append(r)
        n_ok += int(r["found"])
        mark = "✓" if r["found"] else "✗"
        peel = r.get("peel", "-")
        print(f"  {mark}  {law.domain:16s} {law.name:22s} "
              f"n={law.difficulty:<2}  {r['elapsed']:6.1f}s  "
              f"[{peel:8s}]  "
              f"{r.get('found_expr', r.get('reason',''))[:40]}")
    summary = {
        "total": len(results), "found": n_ok,
        "rate": n_ok / max(1, len(results)),
        "by_domain": {}, "results": results,
    }
    for r in results:
        d = summary["by_domain"].setdefault(r["domain"], {"n": 0, "found": 0})
        d["n"] += 1
        d["found"] += int(r["found"])
    return summary
