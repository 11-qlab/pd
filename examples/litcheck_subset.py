"""Fast validation of litcheck against 12 canonical laws.

v.confidence is a STRING in current litcheck (e.g. "high", "0.85"),
not a float — this script coerces safely either way.
"""
from purediscover.litcheck import litcheck
from purediscover.resource.laws import get

SUBSET = [
    ("newton_2",            "F = m·a"),
    ("emc2",                "E = m·c²"),
    ("photon_energy",       "E = h·f"),
    ("ohm",                 "V = I·R"),
    ("kinetic_energy",      "E = ½·m·v²"),
    ("gravitational_force", "F = G·m₁·m₂/r²"),
    ("coulomb",             "F = k·q₁·q₂/r²"),
    ("ideal_gas",           "P·V = n·R·T"),
    ("michaelis_menten",    "v = V·S/(K_m + S)"),
    ("hydrogen_levels",     "E_n = −13.6/n²"),
    ("stefan_boltzmann",    "j = σ·T⁴"),
    ("bayes",               "P(A|B) = P(B|A)·P(A)/P(B)"),
]

KNOWN = {"EXACT_IN_INDEX", "STRONG_LEXICAL_MATCH", "FAMILY_IN_INDEX"}


def _conf_str(conf) -> str:
    """Coerce confidence to a fixed-width string regardless of type."""
    try:
        return f"{float(conf):.2f}"
    except (TypeError, ValueError):
        return str(conf)[:6]


print(f"{'':2s} {'law':22s}  {'verdict':22s}  {'conf':>6s}  top paper")
print("=" * 88)
n_known = 0
for name, target in SUBSET:
    law = get(name)
    try:
        v = litcheck(law.expr,
                     var_descriptions=law.variables,
                     limit_per_source=3,
                     ledger_path="litcheck_ledger.jsonl")
    except Exception as e:
        print(f"  ✗ {name:22s}  {'ERROR':22s}  {'-':>6s}  {str(e)[:40]}")
        continue
    known = v.verdict in KNOWN
    n_known += int(known)
    top = v.top_papers[0] if getattr(v, "top_papers", None) else None
    title = (top.title[:45] if top else "")
    mark = "✓" if known else "✗"
    print(f"  {mark} {name:22s}  {v.verdict:22s}  "
          f"{_conf_str(v.confidence):>6s}  {title}")
print("=" * 88)
print(f"  {n_known}/{len(SUBSET)}  "
      f"({n_known/len(SUBSET)*100:.0f}%)")
print()
if n_known >= 10:
    print("  HEALTHY: litcheck returns known-paper verdicts on canonical laws.")
    print("  Proceed to full corpus validation (examples/litcheck_full.py).")
elif n_known >= 7:
    print("  PARTIAL: some canonical laws return NOT_FOUND.")
    print("  Inspect failures above; check litcheck.py query expansion.")
else:
    print("  BROKEN: canonical laws returning NOT_FOUND.")
    print("  Fix litcheck source lookup or query construction first.")
