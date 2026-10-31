"""Resumable full-corpus litcheck, v2.

Two fixes over v1:
    1. RETRIABLE-FAILURE CACHE POLICY
       v1 cached every result including "sources were down" — which
       meant the resumable runner never retried rate-limited laws.

       v2 caches only definitive outcomes:
         - known=True                          -> cache (real witness)
         - sources_failed == []                -> cache (definitive
                                                   analogy or
                                                   contradiction)
       Results with sources_failed != [] are NOT cached, so re-run
       retries them. This is what makes the pattern actually resumable
       across rate-limit cascades.

    2. COUNTER DISPLAY
       v1's `done` counter only incremented on witness. v2 tracks
       processed / witnessed / cached separately so the summary
       reflects reality.

Usage:
    python examples/litcheck_full_resumable.py            # resume
    python examples/litcheck_full_resumable.py --force    # remeasure
    python examples/litcheck_full_resumable.py --retry-inconclusive
        # retry laws whose cached result was NOT_FOUND/ANALOGOUS with
        # some source failures — useful after fixing the cache policy
"""
import argparse
import json
import pathlib
import sys
import time

from purediscover.litcheck_named import assess_law
from purediscover.resource.laws import REGISTRY

OUT = pathlib.Path("snapshots/litcheck_full")
OUT.mkdir(parents=True, exist_ok=True)


def _one(law) -> dict:
    try:
        a = assess_law(law.name, limit_per_source=3)
        return {
            "name": law.name,
            "domain": law.domain,
            "difficulty": law.difficulty,
            "verdict": a["verdict"],
            "confidence": a["confidence"],
            "known": a["known"],
            "promoted": a["promoted"],
            "promotion_reason": a["promotion_reason"],
            "matched_query": a["matched_query"],
            "witness_title": a["witness_title"],
            "strategy": a["strategy"],
            "top_title": a["top_paper"].title if a["top_paper"] else None,
            "top_doi": a["top_paper"].doi if a["top_paper"] else None,
            "sources_ok": a.get("sources_ok", []),
            "sources_failed": a.get("sources_failed", []),
            "error": None,
        }
    except Exception as e:
        return {
            "name": law.name,
            "domain": law.domain,
            "difficulty": law.difficulty,
            "error": f"{type(e).__name__}: {e}",
            "known": False,
            "promoted": False,
            "sources_failed": ["__exception__"],
        }


def _is_retriable(rec: dict) -> bool:
    """True iff this result should NOT be cached (retry next run).

    Policy v2: cache if at least ONE source returned data. s2 is
    chronically dead (fails 100% of calls), so requiring "no source
    failed" left 7 laws in a permanent retry loop even though
    crossref reliably returned titles for them.

    Cache now on:
      - known == True                          (witness)
      - sources_ok has >= 1 entry              (partial evidence)
      - sources_failed == []                   (fully healthy)

    Retry only when:
      - exception during call
      - known == False AND sources_ok == []    (no data at all)
    """
    if rec.get("error"):
        return True
    if rec.get("known"):
        return False
    if len(rec.get("sources_ok") or []) >= 1:
        return False
    if not rec.get("sources_failed"):
        return False
    return True


def _should_skip(out: pathlib.Path, retry_inconclusive: bool) -> bool:
    if not out.exists():
        return False
    try:
        rec = json.loads(out.read_text())
    except Exception:
        return False
    if rec.get("error") and _is_retriable(rec):
        return False
    if retry_inconclusive and _is_retriable(rec):
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--retry-inconclusive", action="store_true")
    ap.add_argument("--only", default="")
    args = ap.parse_args()

    laws = list(REGISTRY)
    if args.only:
        wanted = {s.strip() for s in args.only.split(",") if s.strip()}
        laws = [l for l in laws if l.name in wanted]

    t0 = time.time()
    n_processed = n_witness = n_skipped = n_errored = 0

    for i, law in enumerate(laws, 1):
        out = OUT / f"{law.name}.json"

        if not args.force and _should_skip(out, args.retry_inconclusive):
            n_skipped += 1
            try:
                rec = json.loads(out.read_text())
                mark = "✓" if rec.get("known") else "✗"
                title = (rec.get("witness_title")
                         or rec.get("top_title") or "")[:45]
                print(f"  [{i:>2}/{len(laws)}] {mark}  {law.name:22s}  "
                      f"{rec.get('verdict','-')!s:22s}  {title}  (cached)")
            except Exception:
                print(f"  [{i:>2}/{len(laws)}] =  {law.name:22s}  cached")
            continue

        try:
            r = _one(law)
            if _is_retriable(r):
                # Don't write — will be retried on next run
                mark = "!" if r.get("error") else "↻"
                if r.get("error"):
                    n_errored += 1
                n_processed += 1
                top = (r.get("witness_title") or r.get("top_title")
                       or r.get("error", ""))[:45]
                sources_failed = r.get("sources_failed") or []
                print(f"  [{i:>2}/{len(laws)}] {mark}  {law.name:22s}  "
                      f"{r.get('verdict','-')!s:22s}  {top}  "
                      f"[retriable: {sources_failed}]")
            else:
                out.write_text(json.dumps(r, indent=2))
                n_processed += 1
                if r.get("known"):
                    n_witness += 1
                    mark = "✓"
                    promo = "P" if r.get("promoted") else " "
                else:
                    mark = "✗"
                    promo = " "
                top = (r.get("witness_title")
                       or r.get("top_title") or "")[:45]
                print(f"  [{i:>2}/{len(laws)}] {mark}{promo}  "
                      f"{law.name:22s}  {r.get('verdict','-')!s:22s}  "
                      f"{top}")
        except Exception as e:
            n_errored += 1
            print(f"  [{i:>2}/{len(laws)}] !   {law.name:22s}  "
                  f"outer error: {e}")

    # --- aggregate over cached files only ---
    print()
    print("=" * 96)
    files = list(OUT.glob("*.json"))
    entries = [json.loads(f.read_text()) for f in files]
    entries = [e for e in entries if e.get("name")]

    states = {"WITNESS": 0, "INCONCLUSIVE": 0, "CONTRADICTS": 0, "ERROR": 0}
    for e in entries:
        if e.get("error"):
            states["ERROR"] += 1
        elif e.get("known"):
            states["WITNESS"] += 1
        elif (e.get("verdict") == "NOT_FOUND_IN_INDEX"
              and not e.get("sources_failed")):
            states["CONTRADICTS"] += 1
        else:
            states["INCONCLUSIVE"] += 1

    by_dom: dict[str, dict] = {}
    for e in entries:
        d = by_dom.setdefault(e["domain"], {"n": 0, "witness": 0, "promoted": 0})
        d["n"] += 1
        if e.get("known"):
            d["witness"] += 1
        if e.get("promoted"):
            d["promoted"] += 1

    print(f"  elapsed     : {time.time()-t0:.0f}s  "
          f"(processed={n_processed} witnessed={n_witness} "
          f"cached={n_skipped} errored={n_errored})")
    print(f"  total cached: {len(entries)}/{len(REGISTRY)}")
    retriable = [e["name"] for e in entries
                 if not e.get("known") and e.get("sources_failed")]
    if retriable:
        print(f"  retriable   : {len(retriable)} laws need re-run "
              f"(sources were down)")
    for s, n in states.items():
        print(f"  {s:12s}: {n}")
    print()
    print("  by domain:")
    for d, x in sorted(by_dom.items()):
        print(f"    {d:18s} {x['witness']}/{x['n']}  (promoted {x['promoted']})")

    misses = [e["name"] for e in entries if not e.get("known")]
    if misses:
        print(f"\n  misses ({len(misses)}): {misses}")
    print("=" * 96)

    if states["CONTRADICTS"] > 0:
        print(f"\n  ⚠ {states['CONTRADICTS']} CONTRADICTS — investigate before training:")
        for e in entries:
            if (not e.get("error") and not e.get("known")
                and e.get("verdict") == "NOT_FOUND_IN_INDEX"
                and not e.get("sources_failed")):
                print(f"    - {e['name']:22s}  "
                      f"top: {(e.get('top_title') or '')[:50]}")
        sys.exit(1)


if __name__ == "__main__":
    main()
