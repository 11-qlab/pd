"""Flywheel integration hook.

Two bugs fixed since the first draft:
    1. Test expressions must survive sympy auto-simplification.
       `x + x + x` becomes `3*x` before we count nodes. Use distinct
       terms (a*b + c*d + e*f) so node count reflects the AST.
    2. Cache key must include the expression itself, not just node
       count. Two different expressions with the same node count
       would otherwise share a cache entry.

Cache key: (law_name, str(expr), rounded_fit_error). str(expr) is
correct here — sympy prints deterministically, and the cost is
negligible compared to litcheck network calls.

If you need to bypass the cache (e.g. after adding a new source),
construct a fresh hook via make_audit_hook() — caches are per-hook.

corpus_only=True: novel laws (law_name=None) are skipped (hook
returns None). Set corpus_only=False once you want to allow novel
candidate discoveries through with node+fit gating only.
"""
from __future__ import annotations
from typing import Any, Callable

from purediscover.audit import audit_discovery, FullAuditResult


def _count_nodes(expr) -> int:
    """Symbolic node count. Non-sympy -> 1."""
    try:
        import sympy as sp
        if not isinstance(expr, sp.Basic):
            return 1
        return sum(1 for _ in sp.preorder_traversal(expr))
    except Exception:
        return 1


def make_audit_hook(
    *,
    max_nodes: int = 8,
    tol: float = 0.02,
    limit_per_source: int = 3,
    corpus_only: bool = True,
    on_accept: Callable[[Any, Any, FullAuditResult], None] | None = None,
    on_reject: Callable[[Any, Any, FullAuditResult], None] | None = None,
) -> Callable[[Any, Any, float, str | None], FullAuditResult | None]:
    """Build a rollout filter callable.

    Returns:
        hook(task, expr, fit_error, law_name) -> FullAuditResult | None
            None is returned only when corpus_only=True AND law_name is
            None (novel law, caller decides what to do).
    """
    cache: dict[tuple[str, str, float], FullAuditResult] = {}

    def hook(task, expr, fit_error: float,
             law_name: str | None) -> FullAuditResult | None:

        if law_name is None and corpus_only:
            return None

        # Cache key includes expr string — distinct expressions never
        # collide even with identical node_count and fit_error.
        expr_str = str(expr)
        key = (law_name or "__novel__", expr_str, round(fit_error, 4))

        if key in cache:
            result = cache[key]
        else:
            n_nodes = _count_nodes(expr)
            if law_name is None:
                result = FullAuditResult(
                    accept=(n_nodes <= max_nodes and fit_error <= tol),
                    lit=None,  # type: ignore[arg-type]
                    node_count=n_nodes,
                    consistency_error=fit_error,
                    max_nodes=max_nodes, tol=tol,
                    reasons=["novel_law_no_litcheck"],
                )
            else:
                result = audit_discovery(
                    law_name=law_name,
                    node_count=n_nodes,
                    consistency_error=fit_error,
                    max_nodes=max_nodes, tol=tol,
                    limit_per_source=limit_per_source,
                )
            cache[key] = result

        if result.accept and on_accept is not None:
            on_accept(task, expr, result)
        elif not result.accept and on_reject is not None:
            on_reject(task, expr, result)
        return result

    return hook


# --- smoke test ---
if __name__ == "__main__":
    import sympy as sp
    from purediscover.resource.laws import get

    print("Flywheel audit hook — smoke test")
    print("=" * 72)

    hook = make_audit_hook()

    # Canonical 3-node target
    law = get("kinetic_energy")
    expr_ok = sp.Symbol("m") * sp.Symbol("v") ** 2

    # Fat expression: DISTINCT terms so sympy doesn't collapse.
    # a*b + c*d + e*f  ->  1 (Add) + 3*3 (three Muls with symbols) = 10 nodes
    a, b, c, d, e, f = sp.symbols("a b c d e f")
    expr_fat = a * b + c * d + e * f
    n_fat = _count_nodes(expr_fat)
    print(f"  fat expression node count: {n_fat} (expected >8)")
    assert n_fat > 8, f"fat expr only has {n_fat} nodes"
    assert _count_nodes(expr_ok) <= 8, "canonical expr should fit"

    # Case 1: corpus law, within node budget, zero fit error -> accept
    r = hook(task=None, expr=expr_ok, fit_error=0.0, law_name=law.name)
    print(f"  case 1  {law.name:22s}  accept={r.accept}  "
          f"nodes={r.node_count}  state={r.lit.state.value}  "
          f"reasons={r.reasons}")
    assert r.accept, "case 1: expected accept"

    # Case 2: node count over budget -> reject
    r = hook(task=None, expr=expr_fat, fit_error=0.0, law_name=law.name)
    print(f"  case 2  {'node_overflow':22s}  accept={r.accept}  "
          f"nodes={r.node_count}  reasons={r.reasons}")
    assert not r.accept, f"case 2: expected reject (nodes={r.node_count})"

    # Case 3: fit error over tolerance -> reject
    r = hook(task=None, expr=expr_ok, fit_error=0.5, law_name=law.name)
    print(f"  case 3  {'fit_error_high':22s}  accept={r.accept}  "
          f"fit_error={r.consistency_error}  reasons={r.reasons}")
    assert not r.accept, "case 3: expected reject"

    # Case 4: novel law with corpus_only=True -> skip (None)
    r = hook(task=None, expr=expr_ok, fit_error=0.0, law_name=None)
    print(f"  case 4  {'novel_law_skip':22s}  "
          f"result={'None (skipped)' if r is None else r}")
    assert r is None, "case 4: expected skip"

    # Case 5: two distinct exprs with same node count, distinct cache
    expr_a = sp.Symbol("x") * sp.Symbol("y")           # 3 nodes
    expr_b = sp.Symbol("p") * sp.Symbol("q")           # 3 nodes
    ra = hook(task=None, expr=expr_a, fit_error=0.0, law_name="kinetic_energy")
    rb = hook(task=None, expr=expr_b, fit_error=0.0, law_name="kinetic_energy")
    same_obj = ra is rb
    print(f"  case 5  {'cache_isolation':22s}  "
          f"same_object={same_obj}  (expect False)")
    assert not same_obj, "case 5: cache must not share across distinct exprs"

    print()
    print("  ✓ hook works. Copy make_audit_hook() into your training code.")
    print("    The audit gate has been exercised on: accept, node reject,")
    print("    fit reject, skip, and cache isolation.")
