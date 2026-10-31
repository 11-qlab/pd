#!/usr/bin/env python3
"""Stateful batch discovery.

Layout:
    datasets/input/        drop *.csv here
    datasets/known/        discovered laws, one JSON per source CSV
                           (each file holds a list of discoveries)
    datasets/run_log.jsonl append-only log

Behavior:
    - process new or edited CSVs
    - skip unchanged CSVs (mtime match)
    - NOVEL forms checked against *all* known laws at the data range
    - on save, merge into existing discoveries (no duplicates)
    - "no changes" rounds still update mtime so they stay skipped
"""
from __future__ import annotations
import csv as _csv
import json
import math
import pathlib
from datetime import datetime, timezone

import sympy as sp

from purediscover.discover import load_csv, discover_and_classify
from purediscover.resource.laws import Law, REGISTRY


INPUT_DIR = pathlib.Path("datasets/input")
KNOWN_DIR = pathlib.Path("datasets/known")
RUN_LOG   = pathlib.Path("datasets/run_log.jsonl")

TARGET_CANDIDATES = ["y", "target", "output", "P", "V", "N", "k"]
MATCH_TOLERANCE   = 0.05


# --- storage -----------------------------------------------------------

def ensure_dirs():
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    KNOWN_DIR.mkdir(parents=True, exist_ok=True)


def load_known() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for f in sorted(KNOWN_DIR.glob("*.json")):
        try:
            out[f.stem] = json.loads(f.read_text())
        except Exception:
            continue
    return out


def save_known(stem: str, record: dict):
    path = KNOWN_DIR / f"{stem}.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(record, indent=2))
    tmp.replace(path)


def append_run_log(entry: dict):
    with RUN_LOG.open("a") as f:
        f.write(json.dumps(entry) + "\n")


# --- FIXED: laws_from_stored iterates "discovered" array ---------------

def _laws_from_stored(stem: str, rec: dict) -> list[Law]:
    """Return all laws saved under this file. Handles the saved format
    where each discovery lives under rec["discovered"][i]["expr"]."""
    laws: list[Law] = []
    discovered = rec.get("discovered") or []

    # Backwards compat: some very old records may have flat "expr"
    legacy = rec.get("expr")
    if legacy and not discovered:
        discovered = [{"expr": legacy}]

    for i, d in enumerate(discovered):
        expr_str = d.get("expr") if isinstance(d, dict) else d
        if not expr_str:
            continue
        try:
            laws.append(Law(
                name=f"discovered__{stem}__{i}",
                domain="discovered",
                expr=sp.sympify(expr_str),
                full_form=expr_str,
                variables={v: v for v in rec.get("variables", [])},
                units="?",
                source=f"discovered from {rec.get('source_csv', '?')}",
                difficulty=int(d.get("n_nodes", 5)) if isinstance(d, dict) else 5,
                ops_needed=(),
            ))
        except Exception:
            continue
    return laws


def build_stored_laws(known: dict[str, dict]) -> list[Law]:
    out: list[Law] = []
    for stem, rec in known.items():
        out.extend(_laws_from_stored(stem, rec))
    return out


# --- data-range matching against stored laws --------------------------

def _matches_stored(expr, data, stored_laws,
                    atol: float = MATCH_TOLERANCE) -> str | None:
    """Data-range-aware: is `expr` a scaled version of a stored law?"""
    for law in stored_laws:
        ratios: list[float] = []
        log_diffs: list[float] = []
        for env, _ in data:
            try:
                a = float(expr.subs({sp.Symbol(k): v for k, v in env.items()}))
                b = float(law.expr.subs({sp.Symbol(k): v for k, v in env.items()}))
            except Exception:
                continue
            if not (math.isfinite(a) and math.isfinite(b)):
                continue
            if abs(b) > 1e-200:
                ratios.append(a / b)
            if a > 1e-200 and b > 1e-200:
                log_diffs.append(math.log(a) - math.log(b))
        if len(ratios) >= len(data) // 2:
            r0 = ratios[0]
            if all(abs(r - r0) <= atol * max(1.0, abs(r0)) for r in ratios):
                return law.name
        if len(log_diffs) >= len(data) // 2:
            d0 = log_diffs[0]
            if all(abs(d - d0) <= atol * max(1.0, abs(d0)) for d in log_diffs):
                return law.name
    return None


# --- per-file processing -----------------------------------------------

def find_target(headers: list[str]) -> str | None:
    return next((c for c in TARGET_CANDIDATES if c in headers), None)


def process_csv(csv_path: pathlib.Path, known: dict) -> dict:
    stem = csv_path.stem
    mtime = csv_path.stat().st_mtime
    prev = known.get(stem)

    if prev and abs(prev.get("source_mtime", 0) - mtime) < 0.001:
        return {"csv": csv_path.name, "status": "skip_unchanged",
                "n_novel": 0,
                "note": f"{prev.get('n_discovered', 0)} law(s) already saved"}

    with csv_path.open(newline="") as f:
        headers = _csv.DictReader(f).fieldnames or []
    target_col = find_target(headers)
    if target_col is None:
        return {"csv": csv_path.name, "status": "skip_no_target",
                "n_novel": 0, "note": f"no numeric target in {headers}"}

    try:
        data, var_names, target = load_csv(csv_path, target_col=target_col)
    except Exception as e:
        return {"csv": csv_path.name, "status": "load_error",
                "n_novel": 0, "note": str(e)}

    if len(data) < 3:
        return {"csv": csv_path.name, "status": "skip_too_few_rows",
                "n_novel": 0, "note": f"{len(data)} rows"}

    stored_laws = build_stored_laws(known)
    try:
        results = discover_and_classify(
            data, var_names,
            corpus=REGISTRY + stored_laws,
            time_budget_direct=30,
            time_budget_compose=90,
        )
    except Exception as e:
        return {"csv": csv_path.name, "status": "discovery_error",
                "n_novel": 0, "note": f"{type(e).__name__}: {e}"}

    if not results:
        return {"csv": csv_path.name, "status": "no_candidates",
                "n_novel": 0, "note": ""}

    discoveries = []
    for r in results:
        if r["class"] != "NOVEL":
            continue
        match = _matches_stored(r["expr"], data, stored_laws)
        if match:
            continue
        discoveries.append(r)

    # nothing NEW relative to what's already stored for this file
    existing = prev.get("discovered", []) if prev else []
    existing_exprs = {d["expr"] for d in existing}

    novel_entries = []
    for r in discoveries:
        expr_str = str(sp.simplify(r["expr"]))
        if expr_str in existing_exprs:
            continue
        novel_entries.append({
            "expr": expr_str,
            "fit_error": r["fit_error"],
            "n_nodes": r["n_nodes"],
            "saved_at": datetime.now(timezone.utc).isoformat(),
        })

    if not novel_entries:
        # No new laws; still update mtime so this file stops re-running
        if prev:
            prev["source_mtime"] = mtime
            save_known(stem, prev)
            return {"csv": csv_path.name, "status": "no_new",
                    "n_novel": 0,
                    "note": f"nothing new; {len(existing)} law(s) kept"}
        return {"csv": csv_path.name, "status": "no_new",
                "n_novel": 0, "note": ""}

    merged = existing + novel_entries
    record = {
        "source_csv": csv_path.name,
        "source_mtime": mtime,
        "n_rows": len(data),
        "variables": var_names,
        "target": target,
        "n_discovered": len(merged),
        "discovered": merged,
    }
    save_known(stem, record)
    return {"csv": csv_path.name, "status": "saved",
            "n_novel": len(novel_entries),
            "discoveries": [d["expr"] for d in novel_entries]}


# --- main loop ---------------------------------------------------------

_MARK = {
    "saved":             "★",
    "skip_unchanged":    "=",
    "skip_no_target":    "?",
    "skip_too_few_rows": "?",
    "load_error":        "!",
    "discovery_error":   "!",
    "no_candidates":     "-",
    "no_new":            "-",
}


def run_batch() -> list[dict]:
    ensure_dirs()
    csvs = sorted(INPUT_DIR.glob("*.csv"))
    if not csvs:
        print(f"  (no CSVs in {INPUT_DIR}/)")
        return []

    print(f"  processing {len(csvs)} CSV(s) from {INPUT_DIR}/\n")
    known = load_known()
    summaries: list[dict] = []

    for csv_path in csvs:
        summary = process_csv(csv_path, known)
        summaries.append(summary)
        if summary["status"] in ("saved", "no_new"):
            known = load_known()

        mark = _MARK.get(summary["status"], "?")
        line = f"  {mark} {summary['csv']:35s} [{summary['status']}]"
        if summary.get("note"):
            line += f"  {summary['note']}"
        print(line)
        for expr in summary.get("discoveries", []):
            print(f"      → {expr}")

    append_run_log({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_csvs": len(csvs),
        "n_saved": sum(1 for s in summaries if s["status"] == "saved"),
        "n_novel": sum(s.get("n_novel", 0) for s in summaries),
        "summaries": summaries,
    })

    print()
    new_files = sum(1 for s in summaries if s["status"] == "saved")
    new_laws  = sum(s.get("n_novel", 0) for s in summaries)
    n_known_laws = sum(len(r.get("discovered", [])) for r in load_known().values())
    print(f"  {new_files} CSV(s) yielded new, {new_laws} law(s) total")
    print(f"  known: {n_known_laws} law(s) across "
          f"{len(load_known())} file(s) in {KNOWN_DIR}/")
    return summaries


def main():
    print("Stateful batch discovery")
    print("=" * 65)
    print(f"  input : {INPUT_DIR}/")
    print(f"  known : {KNOWN_DIR}/")
    print(f"  log   : {RUN_LOG}")
    print("=" * 65)

    round_n = 0
    while True:
        round_n += 1
        print(f"\n--- round {round_n} ---\n")
        run_batch()
        print()
        try:
            ans = input("Run again? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if ans not in ("y", "yes"):
            print("Bye.")
            break


if __name__ == "__main__":
    main()
