"""Literature check for discovered expressions.

What this does: given a sympy expression and human-readable descriptions
of its variables, query OpenAlex / Semantic Scholar / arXiv / Crossref /
zbMATH, rerank results lexically and structurally, and return a graded
verdict plus a ledger entry.

What this does NOT do: prove novelty. Every verdict is evidence of
absence-in-index, not absence-in-world. Abstracts are indexed; Section 4
of a PDF is not. Formulas inside PDF images are not searchable at all.
"""
from __future__ import annotations

import json
import re
import time
import hashlib
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import quote

import requests
import sympy as sp

USER_AGENT = "purediscover/0.1 (litcheck; mailto:you@example.com)"
TIMEOUT = 20

# --------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------

@dataclass
class Paper:
    source: str
    id: str
    title: str
    abstract: str = ""
    year: int | None = None
    doi: str | None = None
    url: str | None = None
    cited_by: int = 0
    score: float = 0.0
    matched_terms: list[str] = field(default_factory=list)
    formula_match: str | None = None   # "exact" | "equivalent" | None


@dataclass
class Verdict:
    expr: str
    verdict: str
    confidence: str
    top_papers: list[Paper]
    query_forms: list[str]
    fingerprint: dict
    sources_ok: list[str]
    sources_failed: list[str]
    timestamp: str
    notes: str = ""

    def to_jsonl(self) -> str:
        return json.dumps(asdict(self), default=str)

# --------------------------------------------------------------------------
# Query construction
# --------------------------------------------------------------------------

_STOP = {"the", "of", "a", "an", "and", "or", "in", "on", "for", "with", "to"}


def _latex(expr: sp.Expr) -> str:
    return sp.latex(expr)


def _infix(expr: sp.Expr) -> str:
    return str(expr)


def _fingerprint(expr: sp.Expr, var_names: list[str]) -> dict:
    fp = {
        "variables": sorted(var_names),
        "node_count": sp.count_ops(expr) + len(expr.free_symbols),
        "ops": sorted({type(a).__name__ for a in sp.preorder_traversal(expr)
                       if isinstance(a, sp.Function) or isinstance(a, sp.Expr)}),
        "latex": _latex(expr),
        "is_polynomial": expr.is_polynomial(*[sp.Symbol(v) for v in var_names]),
    }
    if fp["is_polynomial"] and var_names:
        try:
            poly = sp.Poly(expr, *[sp.Symbol(v) for v in var_names])
            fp["monomials"] = [str(m) for m in poly.monoms()]
            fp["total_degree"] = poly.total_degree()
        except Exception:
            pass
    return fp


def build_queries(
    expr: sp.Expr,
    var_names: list[str],
    var_descriptions: dict[str, str] | None = None,
) -> list[str]:
    """Generate multiple query strings. The natural-language forms are the
    useful ones; the formula forms mostly fail on general engines but do
    catch formula-indexed sources (Approach0, Matlas, arXiv LaTeX)."""
    queries: list[str] = []

    # 1. formula queries (exact LaTeX + infix)
    queries.append(f'"{_latex(expr)}"')
    queries.append(f'"{_infix(expr)}"')

    if not var_descriptions:
        return queries

    # 2. natural-language description of the relation
    desc = [var_descriptions.get(v, v) for v in var_names]
    words = []
    for d in desc:
        words.extend(w for w in re.split(r"\W+", d.lower()) if w and w not in _STOP)
    if words:
        queries.append(" ".join(words))

    # 3. "X times Y" style pairs for products
    if isinstance(expr, sp.Mul):
        factors = [f for f in expr.args if isinstance(f, sp.Symbol)]
        if len(factors) == 2:
            a, b = factors
            da = var_descriptions.get(a.name, a.name)
            db = var_descriptions.get(b.name, b.name)
            queries.append(f"{da} times {db}")
            queries.append(f"{da} {db}")
            queries.append(f"{da} proportional to {db}")

    # 4. ratio forms
    if isinstance(expr, sp.Pow) and expr.exp == -1:
        base = expr.base
        if isinstance(base, sp.Symbol):
            d = var_descriptions.get(base.name, base.name)
            queries.append(f"inverse {d}")
            queries.append(f"proportional to 1 over {d}")

    # dedup
    seen, out = set(), []
    for q in queries:
        if q.lower() not in seen:
            seen.add(q.lower())
            out.append(q)
    return out

# --------------------------------------------------------------------------
# Sources
# --------------------------------------------------------------------------

def _get(url: str, params: dict | None = None) -> dict | None:
    try:
        r = requests.get(url, params=params, timeout=TIMEOUT,
                         headers={"User-Agent": USER_AGENT})
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None


def source_openalex(query: str, limit: int = 8) -> list[Paper]:
    data = _get("https://api.openalex.org/works",
                {"search": query, "per-page": limit,
                 "select": "id,title,abstract_inverted_index,publication_year,doi,cited_by_count"})
    if not data:
        return []
    out = []
    for w in data.get("results", []):
        abstract = _reconstruct_abstract(w.get("abstract_inverted_index"))
        out.append(Paper(
            source="openalex", id=w.get("id", ""), title=w.get("title") or "",
            abstract=abstract, year=w.get("publication_year"),
            doi=(w.get("doi") or "").replace("https://doi.org/", "") or None,
            url=w.get("id"), cited_by=w.get("cited_by_count") or 0,
        ))
    return out


def source_semanticscholar(query: str, limit: int = 8) -> list[Paper]:
    data = _get("https://api.semanticscholar.org/graph/v1/paper/search",
                {"query": query, "limit": limit,
                 "fields": "title,abstract,year,externalIds,citationCount,url"})
    if not data:
        return []
    out = []
    for p in data.get("data", []):
        ext = p.get("externalIds") or {}
        out.append(Paper(
            source="s2", id=p.get("paperId", ""), title=p.get("title") or "",
            abstract=p.get("abstract") or "", year=p.get("year"),
            doi=ext.get("DOI"), url=p.get("url"),
            cited_by=p.get("citationCount") or 0,
        ))
    return out


def source_arxiv(query: str, limit: int = 8) -> list[Paper]:
    url = "https://export.arxiv.org/api/query"
    try:
        r = requests.get(url, params={"search_query": f"all:{query}",
                                      "start": 0, "max_results": limit},
                         timeout=TIMEOUT, headers={"User-Agent": USER_AGENT})
        if r.status_code != 200:
            return []
    except Exception:
        return []
    entries = re.findall(r"<entry>(.*?)</entry>", r.text, re.S)
    out = []
    for e in entries:
        title = re.search(r"<title>(.*?)</title>", e, re.S)
        summary = re.search(r"<summary>(.*?)</summary>", e, re.S)
        link = re.search(r"<id>(.*?)</id>", e, re.S)
        out.append(Paper(
            source="arxiv",
            id=(link.group(1) if link else ""),
            title=(title.group(1).strip() if title else ""),
            abstract=(summary.group(1).strip() if summary else ""),
            url=(link.group(1) if link else None),
        ))
    return out


def source_crossref(query: str, limit: int = 8) -> list[Paper]:
    data = _get("https://api.crossref.org/works",
                {"query": query, "rows": limit, "select": "DOI,title,abstract,published,is-referenced-by-count"})
    if not data:
        return []
    items = (data.get("message") or {}).get("items") or []
    out = []
    for it in items:
        out.append(Paper(
            source="crossref", id=it.get("DOI", ""),
            title=(it.get("title") or [""])[0],
            abstract=it.get("abstract") or "",
            doi=it.get("DOI"),
            cited_by=it.get("is-referenced-by-count") or 0,
        ))
    return out

# --------------------------------------------------------------------------
# Rerank
# --------------------------------------------------------------------------

def _tokens(text: str) -> set[str]:
    return {w for w in re.split(r"\W+", text.lower()) if w and w not in _STOP}


def _reconstruct_abstract(inv: dict | None) -> str:
    if not inv:
        return ""
    pos = {}
    for word, idxs in inv.items():
        for i in idxs:
            pos[i] = word
    return " ".join(pos[i] for i in sorted(pos))


def _lexical_score(paper: Paper, query_terms: set[str]) -> float:
    text = f"{paper.title} {paper.abstract}"
    terms = _tokens(text)
    if not terms or not query_terms:
        return 0.0
    overlap = query_terms & terms
    jac = len(overlap) / len(query_terms | terms)
    paper.matched_terms = sorted(overlap)
    return jac + 0.001 * min(paper.cited_by, 1000) / 1000.0


def _structural_score(paper: Paper, fp: dict) -> float:
    """Try to parse LaTeX formulas out of the abstract and compare."""
    lats = re.findall(r"\$([^$]{3,120})\$", paper.abstract or "")
    if not lats:
        return 0.0
    target_vars = set(fp["variables"])
    best = 0.0
    for lat in lats:
        if not any(v in lat for v in target_vars):
            continue
        try:
            parsed = sp.parsing.latex.parse_latex(lat)
        except Exception:
            continue
        try:
            if sp.simplify(parsed - sp.Symbol("__target__")) == 0:
                continue
        except Exception:
            pass
        if set(s.name for s in parsed.free_symbols) == target_vars:
            best = max(best, 0.3)
            paper.formula_match = "equivalent"
    return best

# --------------------------------------------------------------------------
# Main entry
# --------------------------------------------------------------------------

SOURCES = {
    "openalex": source_openalex,
    "s2": source_semanticscholar,
    "arxiv": source_arxiv,
    "crossref": source_crossref,
}

VERDICT_THRESHOLDS = {
    "exact": 0.60,     # lexical + structural both high
    "family": 0.35,
    "analogous": 0.18,
}


def _verdict(expr: sp.Expr, papers: list[Paper], fp: dict,
             query_forms: list[str], ok: list[str], failed: list[str],
             has_var_descriptions: bool) -> Verdict:
    top = sorted(papers, key=lambda p: p.score, reverse=True)[:10]

    exact = any(p.formula_match == "equivalent" for p in top)
    best = top[0].score if top else 0.0

    if not has_var_descriptions:
        verdict, conf = "INSUFFICIENT_QUERY", "low"
        note = "No variable descriptions supplied. Formula-only queries rarely hit abstracts."
    elif exact:
        verdict, conf = "EXACT_IN_INDEX", "medium"
        note = "Algebraically equivalent formula parsed from an abstract."
    elif best >= VERDICT_THRESHOLDS["exact"]:
        verdict, conf = "STRONG_LEXICAL_MATCH", "medium"
        note = "High term overlap; verify by reading the paper."
    elif best >= VERDICT_THRESHOLDS["family"]:
        verdict, conf = "FAMILY_IN_INDEX", "low"
        note = "Same variables / relation family, different form."
    elif best >= VERDICT_THRESHOLDS["analogous"]:
        verdict, conf = "ANALOGOUS_ONLY", "low"
        note = "Thematic overlap, no structural match."
    else:
        verdict, conf = "NOT_FOUND_IN_INDEX", "low"
        note = "No source returned a high-score candidate. Absence in index, not in world."

    return Verdict(
        expr=str(expr), verdict=verdict, confidence=conf,
        top_papers=top, query_forms=query_forms, fingerprint=fp,
        sources_ok=ok, sources_failed=failed,
        timestamp=datetime.now(timezone.utc).isoformat(),
        notes=note,
    )


def litcheck(
    expr: sp.Expr,
    var_descriptions: dict[str, str] | None = None,
    sources: list[str] | None = None,
    limit_per_source: int = 8,
    ledger_path: str | Path | None = "litcheck_ledger.jsonl",
) -> Verdict:
    var_names = sorted(s.name for s in expr.free_symbols)
    fp = _fingerprint(expr, var_names)
    queries = build_queries(expr, var_names, var_descriptions)
    sources = sources or list(SOURCES.keys())

    # query terms for lexical rerank: union of all natural-language queries
    q_terms: set[str] = set()
    for q in queries:
        q_terms |= _tokens(q)

    all_papers: list[Paper] = []
    ok, failed = [], []

    def run(name, q):
        try:
            return name, SOURCES[name](q, limit_per_source)
        except Exception:
            return name, []

    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = []
        for name in sources:
            # only use natural queries for broad engines; formula queries
            # for arXiv (LaTeX source) and any formula-aware source
            qs = queries if name == "arxiv" else [q for q in queries if not q.startswith('"')]
            if not qs:
                qs = queries[:1]
            for q in qs[:3]:
                futures.append(ex.submit(run, name, q))
        for f in as_completed(futures):
            name, papers = f.result()
            if papers:
                ok.append(name)
                all_papers.extend(papers)
            elif name not in ok:
                failed.append(name)

    # dedup by DOI, then by normalized title
    seen_doi, seen_title, deduped = set(), set(), []
    for p in all_papers:
        if p.doi:
            if p.doi.lower() in seen_doi:
                continue
            seen_doi.add(p.doi.lower())
        t = re.sub(r"\W+", "", p.title.lower())[:80]
        if t and t in seen_title:
            continue
        seen_title.add(t)
        deduped.append(p)

    for p in deduped:
        p.score = _lexical_score(p, q_terms) + _structural_score(p, fp)

    v = _verdict(expr, deduped, fp, queries,
                 sorted(set(ok)), sorted(set(failed)),
                 has_var_descriptions=bool(var_descriptions))

    if ledger_path:
        Path(ledger_path).parent.mkdir(parents=True, exist_ok=True)
        with open(ledger_path, "a", encoding="utf-8") as fh:
            fh.write(v.to_jsonl() + "\n")

    return v
