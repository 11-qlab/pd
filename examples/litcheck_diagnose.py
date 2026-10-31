"""Diagnose litcheck's interface: signature, accepted kwargs, verdict type."""
import inspect
from purediscover.litcheck import litcheck
from purediscover.resource.laws import get

print("litcheck signature:")
print("  ", inspect.signature(litcheck))
print()

law = get("newton_2")
print("baseline call:")
v = litcheck(law.expr, var_descriptions=law.variables,
             limit_per_source=2, ledger_path="litcheck_ledger.jsonl")
print("  verdict   :", repr(v.verdict))
print("  confidence:", repr(v.confidence), type(v.confidence).__name__)
print("  top_papers:", type(v.top_papers).__name__, len(v.top_papers or []))
print()

# What does LitcheckVerdict carry?
print("LitcheckVerdict attrs:",
      [a for a in dir(v) if not a.startswith("_")])
if v.top_papers:
    p = v.top_papers[0]
    print("Paper attrs          :",
          [a for a in dir(p) if not a.startswith("_")])
print()

# Probe: which kwargs does litcheck accept?
print("kwarg probes (newton_2, canonical phrase):")
CANON = "Newton's second law of motion F = ma"
for kw in ("query", "query_hint", "canonical", "name_hint", "context",
           "paper_query", "query_terms", "search_query"):
    try:
        r = litcheck(law.expr, var_descriptions=law.variables,
                     limit_per_source=1, ledger_path="litcheck_ledger.jsonl",
                     **{kw: CANON})
        print(f"  {kw:14s}: accepted  -> {r.verdict:22s}  "
              f"{r.top_papers[0].title[:40] if r.top_papers else ''}")
    except TypeError as e:
        print(f"  {kw:14s}: rejected  ({e})")
    except Exception as e:
        print(f"  {kw:14s}: errored   ({type(e).__name__}: {e})")
print()

# Probe: enriched-description approach (works regardless of kwarg support)
print("enriched-descriptions probe:")
enriched = {k: f"{d} ({CANON})" for k, d in law.variables.items()}
r = litcheck(law.expr, var_descriptions=enriched,
             limit_per_source=2, ledger_path="litcheck_ledger.jsonl")
print(f"  verdict: {r.verdict}  "
      f"top: {r.top_papers[0].title[:50] if r.top_papers else '(none)'}")
