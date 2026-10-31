"""Name-aware litcheck wrapper, v5.

Tries:
    original  var_descriptions = law.variables        (once)
    short:N   each canonical phrase in get_queries()  (in order)
    title:N   title-phrase promotion across all attempts

Short-circuits on the first KNOWN verdict. Promotion (title match)
checked only after all short attempts fail.

Ground truth (2026-10-31):
    formula_match = None on every paper (upstream broken)
    s2 source fails 12/12; arxiv/openalex intermittent
    verdicts are keyword-score only against paper titles
    => title-phrase promotion is the highest-signal check
"""
from __future__ import annotations
import math
import re

from purediscover.litcheck import litcheck
from purediscover.resource.laws import Law, get as get_law
from purediscover.resource.lit_queries import get_queries

KNOWN_VERDICTS = {"EXACT_IN_INDEX", "STRONG_LEXICAL_MATCH", "FAMILY_IN_INDEX"}

_STOPWORDS = {"the", "and", "of", "for", "in", "on", "to", "by", "with",
              "a", "an", "is", "at", "as", "or", "be", "its", "this",
              "that", "from", "into"}

_VERDICT_RANK = {
    "EXACT_IN_INDEX": 5, "STRONG_LEXICAL_MATCH": 4,
    "FAMILY_IN_INDEX": 3, "ANALOGOUS_ONLY": 2, "NOT_FOUND_IN_INDEX": 1,
}
_CONF_RANK = {"high": 3, "medium": 2, "low": 1}


def _tokens(text: str) -> set[str]:
    if not text:
        return set()
    text = text.lower()
    text = re.sub(r"[''`]", "", text)
    return {w for w in re.findall(r"\w+", text)
            if len(w) >= 3 and w not in _STOPWORDS}


def _title_matches(title: str, canonical: str,
                   min_ratio: float = 0.6) -> bool:
    q = _tokens(canonical)
    t = _tokens(title)
    if not q or not t:
        return False
    required = max(2, math.ceil(len(q) * min_ratio))
    return len(q & t) >= required


def _find_title_witness(papers, canonical: str):
    if not papers or not canonical:
        return None, set()
    for p in papers[:5]:
        if _title_matches(p.title or "", canonical):
            return p, _tokens(canonical) & _tokens(p.title or "")
    return None, set()


def _enrich(law: Law, query: str) -> dict[str, str]:
    if not law.variables:
        return {"__query__": query}
    return {k: f"{v} ({query})" for k, v in law.variables.items()}


def _rank(v) -> tuple[int, int]:
    return (_VERDICT_RANK.get(v.verdict, 0),
            _CONF_RANK.get(str(v.confidence).lower(), 0))


def is_known(v) -> bool:
    return v.verdict in KNOWN_VERDICTS


def assess_law(law: Law | str, limit_per_source: int = 3,
               ledger_path: str = "litcheck_ledger.jsonl") -> dict:
    if isinstance(law, str):
        law = get_law(law)

    queries = get_queries(law.name)
    attempts: list[tuple[str, object]] = []

    def _try(label: str, descriptions):
        try:
            v = litcheck(law.expr, var_descriptions=descriptions,
                         limit_per_source=limit_per_source,
                         ledger_path=ledger_path)
            attempts.append((label, v))
            return v
        except Exception:
            return None

    # 1. original
    v = _try("original", law.variables)
    if v is not None and is_known(v):
        return _pack(law, v, "original", attempts, promoted=False)

    # 2. each canonical query
    for i, q in enumerate(queries, 1):
        v = _try(f"short:{i}", _enrich(law, q))
        if v is not None and is_known(v):
            return _pack(law, v, f"short:{i}", attempts, promoted=False)

    # 3. title-phrase promotion across every attempt × every query
    for label, v in attempts:
        for q in queries:
            witness, matched = _find_title_witness(v.top_papers, q)
            if witness is not None:
                return _pack(law, v, label, attempts,
                             promoted=True,
                             promotion_reason="title_phrase_match",
                             matched_tokens=sorted(matched),
                             witness_title=witness.title,
                             matched_query=q)

    if not attempts:
        raise RuntimeError(f"all litcheck attempts failed for {law.name}")
    best_label, best_v = max(attempts, key=lambda t: _rank(t[1]))
    return _pack(law, best_v, best_label, attempts, promoted=False)


def _pack(law: Law, v, strategy: str, attempts, promoted: bool,
          promotion_reason=None, matched_tokens=None, witness_title=None,
          matched_query=None):
    return {
        "name": law.name,
        "verdict": v.verdict,
        "confidence": str(v.confidence),
        "known": is_known(v) or promoted,
        "promoted": promoted,
        "promotion_reason": promotion_reason,
        "matched_tokens": matched_tokens,
        "matched_query": matched_query,
        "witness_title": witness_title,
        "strategy": strategy,
        "top_paper": (v.top_papers[0] if v.top_papers else None),
        "attempts": [(lbl, av.verdict) for lbl, av in attempts],
        "sources_ok": list(getattr(v, "sources_ok", []) or []),
        "sources_failed": list(getattr(v, "sources_failed", []) or []),
        "verdict_obj": v,
    }


def litcheck_named(law: Law | str, limit_per_source: int = 3,
                   ledger_path: str = "litcheck_ledger.jsonl"):
    return assess_law(law, limit_per_source=limit_per_source,
                      ledger_path=ledger_path)["verdict_obj"]


__all__ = ["litcheck_named", "assess_law", "is_known", "KNOWN_VERDICTS"]

# --- enriched ledger: records promotion info so replay_audit can see it ---
import json as _json
from pathlib import Path as _Path

ENRICHED_LEDGER = "purediscover_assess_ledger.jsonl"


def _write_enriched(result: dict, path: str = ENRICHED_LEDGER) -> None:
    """Append one enriched record per assess_law call."""
    try:
        top = result.get("top_paper")
        rec = {
            "name": result["name"],
            "verdict": result["verdict"],
            "confidence": result["confidence"],
            "known": result["known"],
            "promoted": result["promoted"],
            "promotion_reason": result.get("promotion_reason"),
            "matched_query": result.get("matched_query"),
            "witness_title": result.get("witness_title"),
            "strategy": result["strategy"],
            "top_title": top.title if top else None,
            "sources_ok": result.get("sources_ok", []),
            "sources_failed": result.get("sources_failed", []),
        }
        _Path(path).open("a").write(_json.dumps(rec) + "\n")
    except Exception:
        pass  # ledger write is best-effort; never break assessment


# --- patch _pack to call _write_enriched ---
_orig_pack = _pack

def _pack(law, v, strategy, attempts, promoted,
          promotion_reason=None, matched_tokens=None, witness_title=None,
          matched_query=None):
    result = _orig_pack(law, v, strategy, attempts, promoted,
                        promotion_reason=promotion_reason,
                        matched_tokens=matched_tokens,
                        witness_title=witness_title,
                        matched_query=matched_query)
    _write_enriched(result)
    return result


# --- v2 enriched writer: include expr so replay can dedupe by expr ---
def _write_enriched(result, path=ENRICHED_LEDGER):
    """Append enriched record with expr; replaces earlier definition."""
    try:
        v = result.get("verdict_obj")
        expr_str = str(getattr(v, "expr", None)) if v else None
        top = result.get("top_paper")
        rec = {
            "name": result["name"],
            "expr": expr_str,
            "verdict": result["verdict"],
            "confidence": result["confidence"],
            "known": result["known"],
            "promoted": result["promoted"],
            "promotion_reason": result.get("promotion_reason"),
            "matched_query": result.get("matched_query"),
            "witness_title": result.get("witness_title"),
            "strategy": result["strategy"],
            "top_title": top.title if top else None,
            "sources_ok": result.get("sources_ok", []),
            "sources_failed": result.get("sources_failed", []),
        }
        _Path(path).open("a").write(_json.dumps(rec) + "\n")
    except Exception:
        pass
