"""Archive old enriched ledger, repopulate with expr field, verify dedupe."""
import pathlib
import shutil
import subprocess
import sys

EPATH = pathlib.Path("purediscover_assess_ledger.jsonl")
if EPATH.exists():
    arch = EPATH.with_suffix(".jsonl.archive")
    shutil.move(str(EPATH), str(arch))
    print(f"[+] archived old enriched ledger -> {arch}")

print("[*] repopulating enriched ledger via subset_v4 (12 assess_law calls)...")
r = subprocess.run(["python", "examples/litcheck_subset_v4.py"],
                   capture_output=True, text=True)
if r.returncode != 0:
    print(r.stderr)
    sys.exit(1)
for line in r.stdout.splitlines():
    if "known:" in line:
        print(f"    {line.strip()}")

if not EPATH.exists():
    print("✗ enriched ledger not written")
    sys.exit(1)
lines = [ln for ln in EPATH.read_text().splitlines() if ln.strip()]
print(f"    {len(lines)} enriched records")

import json
sample = json.loads(lines[0])
has_expr = "expr" in sample and sample["expr"]
print(f"    first record has expr: {'✓' if has_expr else '✗'}  "
      f"({sample.get('expr')})")
if not has_expr:
    print("✗ expr missing — Script 1 patch didn't take")
    sys.exit(1)

print()
print("[*] per-law replay (expect 12 entries, no duplicates):")
sys.path.insert(0, ".")
from purediscover.audit import replay_audit_per_law, LitState

entries = replay_audit_per_law()
states: dict[str, int] = {}
n_promoted = n_accept = 0
for e in entries:
    states[e["state"]] = states.get(e["state"], 0) + 1
    n_promoted += int(e.get("promoted", False))
    n_accept += int(e.get("accept", False))

print(f"    laws      : {len(entries)}")
print(f"    accepted  : {n_accept}/{len(entries)}")
print(f"    promoted  : {n_promoted}")
print(f"    by state  : {dict(sorted(states.items()))}")
print()
for e in entries:
    ident = e.get("name") or e.get("expr") or "?"
    promo = "P" if e.get("promoted") else " "
    mark = "✓" if e["accept"] else "✗"
    print(f"    {mark} {ident[:30]:30s}  {e['state']:14s}  {promo}")

if len(entries) == 12 and states.get(LitState.WITNESS.value, 0) == 12:
    print("\n  ✓ 12 laws, 12 WITNESS, 0 duplicates, 0 contradictions.")
else:
    print(f"\n  ⚠ unexpected: {len(entries)} entries, "
          f"{states.get(LitState.WITNESS.value, 0)} witness")
