"""Verify replay_audit works and reads the existing ledger.

Two views:
  1. per-attempt  — the raw 156 ledger entries
  2. per-law      — best state per law across all attempts (the metric
                    that matters: did this law ever get witness?)

Also: shows which laws are INCONCLUSIVE / CONTRADICTS, so you know
what still needs tuning vs what's a genuine reject signal.
"""
from purediscover.audit import replay_audit, LitState


def _brief(x, n: int) -> str:
    """String-coerce any value safely for display."""
    if x is None:
        return "?"
    try:
        return str(x)[:n]
    except Exception:
        return "<?>"


def _state_rank(s: str) -> int:
    return {LitState.WITNESS.value: 3,
            LitState.INCONCLUSIVE.value: 2,
            LitState.CONTRADICTS.value: 1}.get(s, 0)


print("Replaying litcheck_ledger.jsonl through 3-state classifier:")
print("=" * 72)

try:
    entries = replay_audit(ledger_path="litcheck_ledger.jsonl")
except Exception as e:
    print(f"  ✗ replay_audit raised: {type(e).__name__}: {e}")
    raise SystemExit(1)

print(f"  entries replayed : {len(entries)}")
if not entries:
    print("  (empty ledger)")
    raise SystemExit(0)

# --- View 1: per-attempt distribution ---
by_state: dict[str, int] = {}
by_verdict: dict[str, int] = {}
for e in entries:
    by_state[e["state"]] = by_state.get(e["state"], 0) + 1
    by_verdict[e["verdict"]] = by_verdict.get(e["verdict"], 0) + 1

accept_n = sum(int(e["accept"]) for e in entries)
print(f"  accepted (attempts) : {accept_n}/{len(entries)}")
print()
print("  by state:")
for s in (LitState.WITNESS.value, LitState.INCONCLUSIVE.value,
          LitState.CONTRADICTS.value):
    print(f"    {s:14s}  {by_state.get(s, 0):>4d}")
print()
print("  by verdict:")
for v in sorted(by_verdict):
    print(f"    {v:24s}  {by_verdict[v]:>4d}")

# --- View 2: per-law best state ---
print()
print("=" * 72)
print("Per-law best state (across all ledger attempts):")
print("=" * 72)

# Group by expr (the ledger key we can rely on, not fingerprint which
# may be dict-shaped). Multiple ledger entries for the same expr are
# attempts at the same law.
by_expr: dict[str, list[dict]] = {}
for e in entries:
    key = _brief(e.get("expr"), 200)
    by_expr.setdefault(key, []).append(e)

law_state: dict[str, str] = {}
law_verdict: dict[str, str] = {}
for expr, es in by_expr.items():
    best = max(es, key=lambda x: _state_rank(x["state"]))
    law_state[expr] = best["state"]
    law_verdict[expr] = best["verdict"]

per_law_state: dict[str, int] = {}
for s in law_state.values():
    per_law_state[s] = per_law_state.get(s, 0) + 1

print(f"  unique exprs in ledger : {len(by_expr)}")
print()
print("  per-law best state:")
for s in (LitState.WITNESS.value, LitState.INCONCLUSIVE.value,
          LitState.CONTRADICTS.value):
    n = per_law_state.get(s, 0)
    pct = n / len(by_expr) * 100 if by_expr else 0
    print(f"    {s:14s}  {n:>3d}  ({pct:.0f}%)")

# Show non-witness laws so you know what tuning remains
print()
non_witness = [(expr, law_state[expr], law_verdict[expr])
               for expr in by_expr
               if law_state[expr] != LitState.WITNESS.value]
if non_witness:
    print(f"  Non-witness exprs ({len(non_witness)}):")
    for expr, state, verdict in sorted(non_witness)[:15]:
        print(f"    {state:14s}  {verdict:22s}  {expr}")
    if len(non_witness) > 15:
        print(f"    ... {len(non_witness) - 15} more")
else:
    print("  ✓ Every expr in the ledger reached WITNESS at least once.")

print("=" * 72)
print()
print("  ✓ replay_audit works. Ledger is re-classifiable.")
print("    This is the audit-report primitive for post-hoc analysis.")
