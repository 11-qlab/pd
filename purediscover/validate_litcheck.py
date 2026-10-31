"""Validate that litcheck returns known-paper verdicts on the law corpus.

The check is: for each law, run litcheck with the law's variable
descriptions. A well-known law should return EXACT_IN_INDEX or
STRONG_LEXICAL_MATCH. NOT_FOUND on a well-known law means the query
expansion or sources are underpowered.
"""
from __future__ import annotations
from purediscover.litcheck import litcheck
from purediscover.resource.laws import Law, REGISTRY


KNOWN_VERDICTS = {"EXACT_IN_INDEX", "STRONG_LEXICAL_MATCH", "FAMILY_IN_INDEX"}


def validate(laws: list[Law] | None = None, limit_per_source: int = 5,
             only: list[str] | None = None) -> dict:
    laws = laws or REGISTRY
    if only:
        laws = [l for l in laws if l.name in only]
    results = []
    n_known = 0
    for law in laws:
        v = litcheck(law.expr, var_descriptions=law.variables,
                     limit_per_source=limit_per_source,
                     ledger_path="litcheck_ledger.jsonl")
        known = v.verdict in KNOWN_VERDICTS
        n_known += int(known)
        top = v.top_papers[0] if v.top_papers else None
        mark = "✓" if known else "✗"
        title = (top.title[:55] if top else "")
        print(f"  {mark}  {law.domain:16s} {law.name:22s}  "
              f"{v.verdict:22s}  {title}")
        results.append({
            "name": law.name,
            "domain": law.domain,
            "verdict": v.verdict,
            "confidence": v.confidence,
            "known": known,
            "top_doi": top.doi if top else None,
        })
    return {
        "total": len(results),
        "known": n_known,
        "rate": n_known / max(1, len(results)),
        "results": results,
    }
