"""Snapshot the full benchmark result to disk for regression tracking.

Usage:
    python examples/snapshot.py            # write snapshots/YYYY-MM-DD.json
    python examples/snapshot.py --diff     # compare to most recent snapshot
"""
from __future__ import annotations
import json
import pathlib
import sys
from datetime import datetime, timezone

from purediscover.peel import run_benchmark_peeled

SNAPDIR = pathlib.Path("snapshots")
SNAPDIR.mkdir(exist_ok=True)


def capture() -> dict:
    summary = run_benchmark_peeled()
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "found": summary["found"],
        "total": summary["total"],
        "rate": summary["rate"],
        "laws": {
            r["name"]: {
                "found": r["found"],
                "peel": r.get("peel"),
                "expr": r.get("found_expr"),
                "elapsed": round(r["elapsed"], 2),
                "domain": r["domain"],
                "difficulty": r["difficulty"],
            }
            for r in summary["results"]
        },
    }


def latest_snapshot() -> pathlib.Path | None:
    snaps = sorted(SNAPDIR.glob("*.json"))
    return snaps[-1] if snaps else None


def diff(new: dict, old: dict) -> int:
    regressions = []
    improvements = []
    for name, new_r in new["laws"].items():
        old_r = old["laws"].get(name)
        if old_r is None:
            continue
        if old_r["found"] and not new_r["found"]:
            regressions.append(name)
        elif not old_r["found"] and new_r["found"]:
            improvements.append(name)
    print()
    print(f"  {old['timestamp']}  ->  {new['timestamp']}")
    print(f"  {old['found']}/{old['total']}  ->  {new['found']}/{new['total']}")
    if improvements:
        print(f"\n  IMPROVED ({len(improvements)}):")
        for name in improvements:
            print(f"    + {name}")
    if regressions:
        print(f"\n  REGRESSED ({len(regressions)}):")
        for name in regressions:
            print(f"    - {name}")
        return 1
    return 0


if __name__ == "__main__":
    print("Capturing benchmark snapshot...")
    new = capture()
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%S")
    path = SNAPDIR / f"{stamp}.json"
    path.write_text(json.dumps(new, indent=2))
    print(f"\nWrote {path}")
    print(f"Found: {new['found']}/{new['total']}  ({new['rate']*100:.1f}%)")

    if "--diff" in sys.argv:
        prev = latest_snapshot()
        if prev and prev != path:
            old = json.loads(prev.read_text())
            sys.exit(diff(new, old))
    sys.exit(0)
