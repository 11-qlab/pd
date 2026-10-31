"""Verify enriched-ledger wiring, populate it, and show the per-law result.

Three phases:
  1. Inspect purediscover/litcheck_named.py for _write_enriched + patched _pack
  2. Call assess_law on the 12 subset laws (writes enriched ledger)
  3. Re-run per-law replay, confirm promoted witnesses surface
"""
import pathlib
import sys

print("PHASE 1: verify litcheck_named.py wiring")
print("=" * 72)
src = pathlib.Path("purediscover/litcheck_named.py").read_text()
checks = {
    "ENRICHED_LEDGER defined":      "ENRICHED_LEDGER =" in src,
    "_write_enriched function":     "def _write_enriched(" in src,
    "_orig_pack captured":          "_orig_pack = _pack" in src,
    "patched _pack wrapper":        "_orig_pack(law, v," in src,
}
ok = True
for label, present in checks.items():
    print(f"  {'✓' if present else '✗'}  {label}")
    if not present:
        ok = False
if not ok:
    print("\n  ✗ wiring incomplete. Re-append Script 1 from previous turn:")
    print("    (the append is idempotent — safe to re-run)")
    sys.exit(1)
print()

print("PHASE 2: populate enriched ledger via subset_v4")
print("=" * 72)
import subprocess
r = subprocess.run(["python", "examples/litcheck_subset_v4.py"],
                   capture_output=True, text=True)
if r.returncode != 0:
    print(r.stderr)
    sys.exit(1)
# just show the summary line, not the full table
for line in r.stdout.splitlines():
    if "known:" in line or "promoted" in line:
        print(f"  {line.strip()}")

enriched_path = pathlib.Path("purediscover_assess_ledger.jsonl")
if not enriched_path.exists():
    print(f"\n  ✗ {enriched_path} not written — assess_law didn't reach _pack")
    sys.exit(1)
n_lines = len([ln for ln in enriched_path.read_text().splitlines() if ln.strip()])
print(f"\n  ✓ enriched ledger: {enriched_path}  ({n_lines} records)")
print()

print("PHASE 3: per-law replay (should now show promoted witnesses)")
print("=" * 72)
from purediscover.audit import replay_audit_per_law, LitState

entries = replay_audit_per_law()
states: dict[str, int] = {}
n_promoted = 0
n_accept = 0
for e in entries:
    states[e["state"]] = states.get(e["state"], 0) + 1
    n_promoted += int(e.get("promoted", False))
    n_accept += int(e.get("accept", False))

print(f"  laws          : {len(entries)}")
print(f"  accepted      : {n_accept}/{len(entries)}")
print(f"  promoted      : {n_promoted}")
print(f"  by state      : {dict(sorted(states.items()))}")
print()
print(f"  {'name/expr':40s}  {'state':14s}  {'promoted':8s}  witness")
print("  " + "-" * 92)
for e in entries:
    ident = e.get("name") or e.get("expr") or "?"
    promo = "P" if e.get("promoted") else " "
    witness = (e.get("witness_title") or "")[:35]
    mark = "✓" if e["accept"] else "✗"
    print(f"  {mark} {ident[:38]:38s}  {e['state']:14s}  {promo:8s}  {witness}")

print()
if states.get(LitState.WITNESS.value, 0) == len(entries):
    print("  ✓ ALL laws WITNESS at per-law level (promotions included).")
    print("    Enriched ledger wiring is correct.")
else:
    misses = [e for e in entries if e["state"] != LitState.WITNESS.value]
    print(f"  ⚠ {len(misses)} non-witness at per-law level:")
    for e in misses:
        ident = e.get("name") or e.get("expr") or "?"
        print(f"    {e['state']:14s}  {ident[:40]}")
