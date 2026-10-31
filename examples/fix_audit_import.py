"""Restore replay_audit into purediscover/audit.py.

Root cause: purediscover/audit.py was overwritten with a 3-state
classifier (audit_discovery, assess_literature, LitState) that omitted
`replay_audit`, which purediscover/__init__.py imports. Package fails
to import.

Order of precedence:
  1. Recover replay_audit from audit.py.bak/.orig/.old if present
  2. Fall back to ledger-based replayer (reads litcheck_ledger.jsonl)

Idempotent. Safe to re-run.
"""
from __future__ import annotations
import pathlib
import re
import shutil
import subprocess

AUDIT = pathlib.Path("purediscover/audit.py")
if not AUDIT.exists():
    raise SystemExit(f"error: {AUDIT} not found; run from project root")

src = AUDIT.read_text()

if "def replay_audit" in src:
    print("[.] audit.py already has replay_audit — nothing to do")
    raise SystemExit(0)

# --- recover from backup if possible ---
recovered = None
for suffix in (".bak", ".orig", ".old", ".v1bak"):
    bak = pathlib.Path(str(AUDIT) + suffix)
    if bak.exists():
        bsrc = bak.read_text()
        if "def replay_audit" in bsrc:
            start = bsrc.index("def replay_audit")
            rest = bsrc[start:]
            m = re.search(r'\n(?=(?:def |class |#\s*===))', rest[1:])
            recovered = rest[:m.start() + 1].rstrip() + "\n" if m else rest.rstrip() + "\n"
            print(f"[+] recovered replay_audit from {bak}")
            break

# --- otherwise, write a compatible implementation ---
if recovered is None:
    print("[.] no backup with replay_audit; writing ledger-based impl")
    recovered = '''def replay_audit(ledger_path: str = "litcheck_ledger.jsonl",
                 limit: int | None = None) -> list[dict]:
    """Replay the audit gate over the litcheck ledger.

    Reads ledger_path (append-only JSONL, one Verdict per line), and
    reclassifies each entry under the current 3-state rule:

        WITNESS      verdict in KNOWN_VERDICTS
        CONTRADICTS  NOT_FOUND_IN_INDEX with all sources healthy
        INCONCLUSIVE otherwise

    Returns a list of dicts in file order:
        {fingerprint, expr, verdict, state, accept}

    Missing ledger returns an empty list.
    """
    import json
    from pathlib import Path

    p = Path(ledger_path)
    if not p.exists():
        return []
    lines = [ln for ln in p.read_text().splitlines() if ln.strip()]
    if limit:
        lines = lines[-limit:]

    out = []
    for line in lines:
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        verdict = e.get("verdict", "NOT_FOUND_IN_INDEX")
        failed = e.get("sources_failed", []) or []
        if verdict in KNOWN_VERDICTS:
            state = LitState.WITNESS
        elif verdict == "NOT_FOUND_IN_INDEX" and not failed:
            state = LitState.CONTRADICTS
        else:
            state = LitState.INCONCLUSIVE
        out.append({
            "fingerprint": e.get("fingerprint"),
            "expr": e.get("expr"),
            "verdict": verdict,
            "state": state.value,
            "accept": state is not LitState.CONTRADICTS,
        })
    return out
'''

# --- append and extend __all__ ---
shutil.copy(AUDIT, str(AUDIT) + ".v2bak")
new_src = src.rstrip() + "\n\n\n" + recovered

m = re.search(r'__all__\s*=\s*\[(.*?)\]', new_src, re.DOTALL)
if m:
    inner = m.group(1)
    if '"replay_audit"' not in inner and "'replay_audit'" not in inner:
        new_inner = inner.rstrip()
        if not new_inner.endswith(","):
            new_inner += ","
        new_inner += '\n    "replay_audit",\n'
        new_src = new_src[:m.start(1)] + new_inner + new_src[m.end(1):]
        print("[+] added 'replay_audit' to audit.py __all__")

AUDIT.write_text(new_src)
print(f"[+] wrote {AUDIT}  (backup: {AUDIT}.v2bak)")

# --- verify import chain ---
result = subprocess.run(
    ["python", "-c", "import purediscover; "
     "from purediscover.audit import replay_audit, audit_discovery, LitState; "
     "print('symbols ok')"],
    capture_output=True, text=True,
)
if result.returncode == 0:
    print("[✓] purediscover imports cleanly:", result.stdout.strip())
else:
    print("[!] import still fails:")
    print(result.stderr)
    print()
    print("  Likely cause: recovered replay_audit references symbols")
    print("  that no longer exist in the new audit.py. Two options:")
    print("    1. Inspect audit.py.v2bak, merge the original body manually")
    print("    2. Guard __init__.py with try/except (see fix script notes)")
