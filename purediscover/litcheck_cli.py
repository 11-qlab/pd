"""CLI: python -m purediscover.litcheck_cli --expr 'a*m' --var a=acceleration --var m=mass"""
import argparse
import json
import sympy as sp
from purediscover.litcheck import litcheck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expr", required=True, help="sympy-parsable expression")
    ap.add_argument("--var", action="append", default=[],
                    help="name=description, repeatable")
    ap.add_argument("--sources", default="openalex,s2,arxiv,crossref")
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--ledger", default="litcheck_ledger.jsonl")
    args = ap.parse_args()

    expr = sp.sympify(args.expr)
    descs = {}
    for v in args.var:
        if "=" not in v:
            raise SystemExit(f"bad --var: {v!r}, expected name=description")
        k, d = v.split("=", 1)
        descs[k.strip()] = d.strip()

    v = litcheck(expr, var_descriptions=descs or None,
                 sources=args.sources.split(","),
                 limit_per_source=args.limit,
                 ledger_path=args.ledger)

    print(f"\nExpression : {v.expr}")
    print(f"Verdict    : {v.verdict}  (confidence: {v.confidence})")
    print(f"Note       : {v.notes}")
    print(f"Sources ok : {v.sources_ok}")
    print(f"Failed     : {v.sources_failed}")
    print(f"Queries    :")
    for q in v.query_forms:
        print(f"  - {q}")
    print(f"\nTop papers:")
    for p in v.top_papers[:8]:
        print(f"  [{p.score:.3f}] {p.source:9s} {p.year or '----'}  {p.title[:80]}")
        if p.matched_terms:
            print(f"           matched: {', '.join(p.matched_terms[:8])}")
        if p.doi:
            print(f"           doi: {p.doi}")
    print(f"\nLedger     : {args.ledger}")


if __name__ == "__main__":
    main()
