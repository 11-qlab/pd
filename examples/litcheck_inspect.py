"""Inspect litcheck internals: query_forms, source health, top-paper scores.

Critical: verdict values seen:
    EXACT_IN_INDEX, STRONG_LEXICAL_MATCH, FAMILY_IN_INDEX  -> witness
    ANALOGOUS_ONLY                                        -> inconclusive
    NOT_FOUND_IN_INDEX                                    -> miss

Also inspects Paper.formula_match and Paper.score — if top paper has
formula_match=True, that paper IS a witness even if verdict is only
ANALOGOUS_ONLY. This is the promotion signal.
"""
from purediscover.litcheck import litcheck
from purediscover.resource.laws import get

SUBSET = [
    "newton_2", "emc2", "photon_energy", "ohm",
    "kinetic_energy", "gravitational_force", "coulomb", "ideal_gas",
    "michaelis_menten", "hydrogen_levels", "stefan_boltzmann", "bayes",
]


def inspect(name: str):
    law = get(name)
    try:
        v = litcheck(law.expr, var_descriptions=law.variables,
                     limit_per_source=3, ledger_path="litcheck_ledger.jsonl")
    except Exception as e:
        print(f"\n  {name:22s}  ERROR  {type(e).__name__}: {e}")
        return

    print(f"\n  {name:22s}  verdict={v.verdict}  conf={v.confidence}")
    print(f"    sources_ok     : {getattr(v, 'sources_ok', '?')}")
    print(f"    sources_failed : {getattr(v, 'sources_failed', '?')}")
    qf = getattr(v, "query_forms", None)
    if qf:
        print(f"    query_forms    :")
        if isinstance(qf, list):
            for q in qf[:5]:
                print(f"       - {str(q)[:80]}")
        elif isinstance(qf, dict):
            for k, val in qf.items():
                print(f"       {k}: {str(val)[:80]}")

    if v.top_papers:
        print(f"    top 3 papers:")
        for i, p in enumerate(v.top_papers[:3], 1):
            fm = getattr(p, "formula_match", None)
            sc = getattr(p, "score", None)
            mt = getattr(p, "matched_terms", None)
            yr = getattr(p, "year", None)
            print(f"      {i}. score={sc}  formula_match={fm}  year={yr}")
            print(f"         title : {p.title[:70] if p.title else ''}")
            if mt:
                print(f"         terms : {mt}")


print("Inspect: query_forms, sources, per-paper formula_match")
print("=" * 80)
for name in SUBSET:
    inspect(name)
print("\n" + "=" * 80)
