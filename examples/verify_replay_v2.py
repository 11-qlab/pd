"""Replay verification, dual view.

Raw (per-attempt)  = forensic; may show CONTRADICTS for individual
                     failed strategy attempts within a successful law
Per-law            = the truth; promoted witnesses visible via
                     enriched ledger; CONTRADICTS only if no attempt
                     anywhere succeeded

Use per-law for any decision. Raw for debugging.
"""
from purediscover.audit import replay_audit, replay_audit_per_law, LitState


def _brief(x, n: int) -> str:
    return str(x)[:n] if x is not None else "?"


def _dist(entries: list[dict], key: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for e in entries:
        out[e[key]] = out.get(e[key], 0) + 1
    return out


print("=" * 72)
print("RAW VIEW (per-attempt; forensic — may include false CONTRADICTS)")
print("=" * 72)
raw = replay_audit(ledger_path="litcheck_ledger.jsonl")
if raw:
    print(f"  attempts      : {len(raw)}")
    print(f"  accepted      : {sum(int(e['accept']) for e in raw)}/{len(raw)}")
    print(f"  by state      : {dict(sorted(_dist(raw, 'state').items()))}")
    print(f"  by verdict    : {dict(sorted(_dist(raw, 'verdict').items()))}")
else:
    print("  (empty)")

print()
print("=" * 72)
print("PER-LAW VIEW (truth — promoted witnesses visible, correct CONTRADICTS)")
print("=" * 72)
per_law = replay_audit_per_law()
if per_law:
    states = _dist(per_law, "state")
    print(f"  laws          : {len(per_law)}")
    print(f"  accepted      : {sum(int(e['accept']) for e in per_law)}/{len(per_law)}")
    print(f"  by state      : {dict(sorted(states.items()))}")
    n_promoted = sum(1 for e in per_law if e.get("promoted"))
    print(f"  promoted      : {n_promoted}")
    print()
    print(f"  {'name/expr':40s}  {'state':14s}  {'verdict':22s}  promoted")
    print("  " + "-" * 88)
    for e in per_law:
        ident = e.get("name") or e.get("expr") or "?"
        mark = "✓" if e["accept"] else "✗"
        promo = "P" if e.get("promoted") else " "
        print(f"  {mark} {ident[:38]:38s}  {e['state']:14s}  "
              f"{(e.get('verdict') or '-'):22s}  {promo}")

    print()
    contradictions = [e for e in per_law
                      if e["state"] == LitState.CONTRADICTS.value]
    if contradictions:
        print(f"  TRUE CONTRADICTS ({len(contradictions)}):")
        for e in contradictions:
            ident = e.get("name") or e.get("expr") or "?"
            print(f"    ✗ {ident[:60]}")
    else:
        print("  ✓ No true contradictions at per-law level.")

print("=" * 72)
