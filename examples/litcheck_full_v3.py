"""Full litcheck v3 validation on all 56 laws. Slower (~5-10 min)."""
import json
import pathlib
import time

from purediscover.litcheck_named import assess_law
from purediscover.resource.laws import REGISTRY


def run():
    results = []
    t0 = time.time()
    for i, law in enumerate(REGISTRY, 1):
        try:
            a = assess_law(law.name, limit_per_source=3)
            results.append({
                "name": law.name, "domain": law.domain,
                "verdict": a["verdict"], "confidence": a["confidence"],
                "known": a["known"], "promoted": a["promoted"],
                "strategy": a["strategy"],
                "top_title": a["top_paper"].title if a["top_paper"] else None,
                "top_doi": a["top_paper"].doi if a["top_paper"] else None,
            })
            mark = "✓" if a["known"] else "✗"
            prom = "P" if a["promoted"] else " "
            top = (a["top_paper"].title[:45] if a["top_paper"] else "")
            print(f"  [{i:>2}/{len(REGISTRY)}] {mark}{prom}  "
                  f"{law.name:22s}  {a['verdict']:22s}  {top}")
        except Exception as e:
            print(f"  [{i:>2}/{len(REGISTRY)}] ✗   {law.name:22s}  ERROR: {e}")
            results.append({
                "name": law.name, "domain": law.domain,
                "verdict": "ERROR", "known": False, "promoted": False,
                "strategy": "-", "error": str(e),
            })

    elapsed = time.time() - t0
    by_domain = {}
    for r in results:
        d = by_domain.setdefault(r["domain"], {"n": 0, "known": 0, "promoted": 0})
        d["n"] += 1
        d["known"] += int(r.get("known", False))
        d["promoted"] += int(r.get("promoted", False))

    summary = {
        "total": len(results),
        "known": sum(r.get("known", False) for r in results),
        "promoted": sum(r.get("promoted", False) for r in results),
        "elapsed": elapsed,
        "by_domain": by_domain,
        "results": results,
    }
    return summary


if __name__ == "__main__":
    print("Full litcheck v3 (multi-attempt + promotion):")
    print("=" * 96)
    s = run()
    print("=" * 96)
    print(f"\nKnown: {s['known']}/{s['total']}  "
          f"({s['known']/s['total']*100:.1f}%)  in {s['elapsed']:.0f}s")
    print(f"Promoted via formula_match: {s['promoted']}")
    print("\nBy domain:")
    for d, x in sorted(s["by_domain"].items()):
        print(f"  {d:18s} {x['known']}/{x['n']}  (promoted {x['promoted']})")

    pathlib.Path("snapshots").mkdir(exist_ok=True)
    stamp = time.strftime("%Y-%m-%dT%H%M%S")
    out = pathlib.Path("snapshots") / f"litcheck_v3_{stamp}.json"
    out.write_text(json.dumps(s, indent=2))
    print(f"\nWrote {out}")
