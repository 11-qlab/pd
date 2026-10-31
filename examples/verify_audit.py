"""Verify audit.py surface, then run subset + audit tests."""
import sys
import traceback

print("STEP 1: import surface")
print("=" * 66)
try:
    from purediscover.audit import (
        audit_discovery, assess_literature, LitState, replay_audit,
    )
    print("  ✓ all expected symbols present:")
    print("      audit_discovery, assess_literature, LitState, replay_audit")
except ImportError as e:
    print(f"  ✗ import failed: {e}")
    sys.exit(1)
print()

print("STEP 2: replay_audit on existing ledger")
print("=" * 66)
try:
    entries = replay_audit(ledger_path="litcheck_ledger.jsonl")
    print(f"  ✓ replayed {len(entries)} ledger entries")
    if entries:
        e = entries[0]
        print(f"    first entry: fingerprint={(e.get('fingerprint') or '?')[:12]}  "
              f"state={e.get('state')}  accept={e.get('accept')}")
except Exception:
    print("  ✗ replay_audit failed:")
    traceback.print_exc()
    print()
print()

print("STEP 3: run subset v4")
print("=" * 66)
import subprocess
r = subprocess.run(["python", "examples/litcheck_subset_v4.py"],
                   capture_output=True, text=True)
print(r.stdout)
if r.returncode != 0:
    print(r.stderr)
    sys.exit(1)
print()

print("STEP 4: run audit_corpus")
print("=" * 66)
r = subprocess.run(["python", "examples/audit_corpus.py"],
                   capture_output=True, text=True)
print(r.stdout)
if r.returncode != 0:
    print(r.stderr)
    sys.exit(1)
