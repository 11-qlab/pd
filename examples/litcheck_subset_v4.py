"""Litcheck subset v4: title-phrase promotion."""
from purediscover.litcheck_named import assess_law

SUBSET = [
    "newton_2", "emc2", "photon_energy", "ohm",
    "kinetic_energy", "gravitational_force", "coulomb", "ideal_gas",
    "michaelis_menten", "hydrogen_levels", "stefan_boltzmann", "bayes",
]

print(f"  {'law':22s}  {'verdict':22s}  {'strat':9s}  {'promo':18s}  top title")
print("=" * 112)
n_known = 0
n_promo = 0
known_list = []
miss_list = []
for name in SUBSET:
    try:
        a = assess_law(name)
    except Exception as e:
        print(f"  ✗ {name:22s}  ERROR: {e}")
        continue
    if a["known"]:
        n_known += 1
        known_list.append(name)
    else:
        miss_list.append(name)
    if a["promoted"]:
        n_promo += 1
    mark = "✓" if a["known"] else "✗"
    promo = a["promotion_reason"] or ""
    witness = a["witness_title"] or (
        a["top_paper"].title if a["top_paper"] else "")
    print(f"  {mark} {name:22s}  {a['verdict']:22s}  {a['strategy']:9s}  "
          f"{promo[:18]:18s}  {(witness or '')[:45]}")

print("=" * 112)
print(f"  known: {n_known}/{len(SUBSET)}  ({n_known/len(SUBSET)*100:.0f}%)  "
      f"promoted: {n_promo}")
print(f"  known : {known_list}")
print(f"  miss  : {miss_list}")
print()
if n_known >= 7:
    print("  STATE: usable. Litcheck is a keyword-only witness detector.")
    print("  ANALOGOUS_ONLY is INCONCLUSIVE — safe for flywheel audit.")
    print("  Proceed to litcheck_full_v4.py.")
else:
    print("  STATE: low. Inspect misses individually — likely no matching")
    print("  title exists in retrieval. Not fixable at this layer.")
