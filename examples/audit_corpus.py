"""Dry-run the flywheel audit gate on the 12-law subset.

For each law: assume the discovery engine found it correctly
(node_count <= 8, consistency_error = 0.0) and ask the audit gate
whether it would accept.
"""
from purediscover.audit import audit_discovery, LitState

SUBSET = [
    "newton_2", "emc2", "photon_energy", "ohm",
    "kinetic_energy", "gravitational_force", "coulomb", "ideal_gas",
    "michaelis_menten", "hydrogen_levels", "stefan_boltzmann", "bayes",
]

print(f"  {'law':22s}  {'state':14s}  {'accept':6s}  reasons")
print("=" * 96)
counts = {s.value: 0 for s in LitState}
n_accept = 0
for name in SUBSET:
    try:
        r = audit_discovery(name, node_count=4, consistency_error=0.0)
    except Exception as e:
        print(f"  ✗ {name:22s}  ERROR: {e}")
        continue
    counts[r.lit.state.value] += 1
    n_accept += int(r.accept)
    mark = "✓" if r.accept else "✗"
    print(f"  {mark} {name:22s}  {r.lit.state.value:14s}  "
          f"{'accept' if r.accept else 'REJECT':6s}  "
          f"{', '.join(r.reasons)}")

print("=" * 96)
print(f"  accepted: {n_accept}/{len(SUBSET)}")
print(f"  WITNESS:      {counts['WITNESS']}")
print(f"  INCONCLUSIVE: {counts['INCONCLUSIVE']}")
print(f"  CONTRADICTS:  {counts['CONTRADICTS']}")
print()
print("  Interpretation:")
print("    WITNESS      -> literature confirms; high-confidence accept")
print("    INCONCLUSIVE -> no clean signal; accept but flag for review")
print("    CONTRADICTS  -> all sources healthy and nothing matched; reject")
