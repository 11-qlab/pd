"""Litcheck subset v3: multi-attempt with promotion."""
from purediscover.litcheck_named import assess_law

SUBSET = [
    "newton_2", "emc2", "photon_energy", "ohm",
    "kinetic_energy", "gravitational_force", "coulomb", "ideal_gas",
    "michaelis_menten", "hydrogen_levels", "stefan_boltzmann", "bayes",
]

print(f"  {'law':22s}  {'verdict':22s}  {'strat':9s}  {'prom':4s}  top")
print("=" * 96)
n_known = 0
n_promoted = 0
for name in SUBSET:
    try:
        a = assess_law(name)
    except Exception as e:
        print(f"  ✗ {name:22s}  ERROR: {e}")
        continue
    n_known += int(a["known"])
    n_promoted += int(a["promoted"])
    mark = "✓" if a["known"] else "✗"
    prom = "✓" if a["promoted"] else " "
    top = a["top_paper"]
    title = (top.title[:45] if top else "")
    print(f"  {mark} {name:22s}  {a['verdict']:22s}  {a['strategy']:9s}  "
          f"{prom:4s}  {title}")

print("=" * 96)
print(f"  {n_known}/{len(SUBSET)}  known   ({n_known/len(SUBSET)*100:.0f}%)")
print(f"  {n_promoted}/{len(SUBSET)}  promoted via formula_match")

if n_known >= 10:
    print("\n  HEALTHY: ready for full corpus validation.")
elif n_known >= 8:
    print("\n  USABLE: acceptable witness rate for flywheel audit.")
    print("  Run litcheck_full_v3.py to see full corpus.")
else:
    print("\n  STILL LOW: inspect remaining failures with litcheck_inspect.py.")
    print("  Consider tuning lit_queries.py per-failure.")
