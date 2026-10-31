"""Best-effort patch of purediscover/symbolic.py:

  1. add 'log' to the default ops tuple if absent
  2. add exponents 4, -1, 1/2 to any explicit exponent list

Backs up to symbolic.py.bak. Prints a unified diff.
Dry-run by default; pass --apply to write.

If a pattern cannot be found, it prints a clear notice — do not run
--apply on a file it says it cannot handle; open the file and patch
by hand using the printed hint.
"""
import difflib
import pathlib
import re
import shutil
import sys

APPLY = "--apply" in sys.argv
p = pathlib.Path("purediscover/symbolic.py")
if not p.exists():
    print(f"error: {p} not found (run from project root)"); sys.exit(1)

orig = p.read_text()
src = orig
changes = []
warnings = []

# --- patch 1: log op ---
if '"log"' in src or "'log'" in src:
    print("[i] 'log' already present in symbolic.py")
else:
    pat_hits = False
    for pat, label in [
        (r'(self\.ops\s*=\s*ops\s+or\s+\()([^)]+)(\))', "self.ops = ops or (...)"),
        (r'(OPS\s*=\s*\()([^)]+)(\))',                 "OPS = (...)"),
        (r'(DEFAULT_OPS\s*=\s*\()([^)]+)(\))',         "DEFAULT_OPS = (...)"),
    ]:
        m = re.search(pat, src)
        if m and "pow" in m.group(2):
            inner = m.group(2).rstrip().rstrip(",")
            src = src[:m.start()] + m.group(1) + inner + ', "log"' + m.group(3) + src[m.end():]
            changes.append(f"added 'log' to ops via {label}")
            pat_hits = True
            break
    if not pat_hits:
        warnings.append(
            "could not locate default ops tuple — open symbolic.py and "
            "append \"log\" to the default ops value passed to Grammar"
        )

# --- patch 2: pow exponents ---
if '"pow2"' in src or "'pow2'" in src:
    warnings.append(
        "Grammar uses separate pow2/pow3 ops — consolidating to a single "
        "'pow' that accepts exponents (2,3,4,-1,1/2) requires a manual "
        "edit of the op dispatch table. See the diff hints printed below."
    )
else:
    extras = []
    pat_hits = False
    for pat in [
        r'(exponents\s*=\s*\[)([^\]]+)(\])',
        r'(POW_EXPONENTS\s*=\s*\[)([^\]]+)(\])',
        r'(exponents\s*=\s*\()([^)]+)(\))',
    ]:
        m = re.search(pat, src)
        if m:
            inner = m.group(2)
            have = {tok.strip() for tok in inner.split(",")}
            for want, lit in [("4", "4"), ("-1", "-1"),
                              ("1/2", "sp.Rational(1, 2)")]:
                if want not in inner and want.replace("/", "") not in have:
                    extras.append(lit)
            if extras:
                src = (src[:m.start()] + m.group(1) + inner
                       + ", " + ", ".join(extras) + m.group(3) + src[m.end():])
                changes.append(f"added pow exponents {extras}")
            pat_hits = True
            break
    if not pat_hits:
        warnings.append(
            "no explicit exponent list found — pow may already be general; "
            "verify with examples/diagnose.py before assuming a change is needed"
        )

# --- report ---
print()
for w in warnings:
    print("[!]", w)
print()
if src == orig:
    print("[.] no changes made")
    sys.exit(0)

print(f"[+] {len(changes)} change(s):")
for c in changes:
    print("    -", c)

print()
for line in difflib.unified_diff(
        orig.splitlines(), src.splitlines(),
        fromfile=str(p), tofile=str(p) + ".patched", lineterm=""):
    print(line)

if APPLY:
    shutil.copy(p, str(p) + ".bak")
    p.write_text(src)
    print(f"\n[+] wrote {p}  (backup: {p}.bak)")
    print("[+] verify: python examples/diagnose.py")
else:
    print("\n[dry-run] pass --apply to write")
