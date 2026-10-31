"""Litcheck subset validation using canonical law-name queries.

Compares plain litcheck (var_descriptions only) vs litcheck_named
(canonical enriched). Shows the exact win per law.
"""
from purediscover.litcheck import litcheck
from purediscover.litcheck_named import litcheck_named, is_known
from purediscover.resource.laws import get

SUBSET = [
    "newton_2", "emc2", "photon_energy", "ohm",
    "kinetic_energy", "gravitational_force", "coulomb", "ideal_gas",
    "michaelis_menten", "hydrogen_levels", "stefan_boltzmann", "bayes",
]


def _conf_str(c) -> str:
    try:
        return f"{float(c):.2f}"
    except (TypeError, ValueError):
        return str(c)[:6]


print(f"{'':3s}{'law':22s}  {'plain':22s}  {'named':22s}")
print("=" * 80)
n_plain = n_named = 0
for name in SUBSET:
    law = get(name)
    try:
        v_plain = litcheck(law.expr, var_descriptions=law.variables,
                           limit_per_source=3, ledger_path="litcheck_ledger.jsonl")
    except Exception as e:
        v_plain = None
        print(f"  {name:22s}  plain errored: {e}")

    try:
        v_named = litcheck_named(law, limit_per_source=3,
                                 ledger_path="litcheck_ledger.jsonl")
    except Exception as e:
        v_named = None
        print(f"  {name:22s}  named errored: {e}")

    plain_known = v_plain is not None and is_known(v_plain.verdict)
    named_known = v_named is not None and is_known(v_named.verdict)
    n_plain += int(plain_known)
    n_named += int(named_known)

    mark_p = "✓" if plain_known else "✗"
    mark_n = "✓" if named_known else "✗"
    verdict_p = v_plain.verdict if v_plain else "-"
    verdict_n = v_named.verdict if v_named else "-"
    print(f"  {mark_p}{mark_n} {name:22s}  {verdict_p:22s}  {verdict_n:22s}")

print("=" * 80)
print(f"  plain: {n_plain}/{len(SUBSET)}   named: {n_named}/{len(SUBSET)}")
print()

if n_named >= 10:
    print("  HEALTHY: canonical queries recover ≥10/12. Litcheck ready for")
    print("  flywheel audit gating. Proceed to litcheck_full_v2.py.")
elif n_named >= 7:
    print("  PARTIAL: improved but some failures. Inspect the ✗ rows —")
    print("  check whether canonical query in lit_queries.py is specific enough.")
else:
    print("  STILL BROKEN: canonical enrichment didn't help. Possible causes:")
    print("    - litcheck's query builder ignores var_descriptions, only uses expr")
    print("    - upstream API (Crossref / OpenAlex) is returning noise")
    print("  Run examples/litcheck_diagnose.py — it probes kwargs directly.")
