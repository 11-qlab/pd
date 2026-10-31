"""Flywheel audit rule.

Three-state literature assessment for a discovered expression:

    WITNESS       verdict in KNOWN_VERDICTS, OR title-phrase match.
                  -> accept with high confidence
    INCONCLUSIVE  ANALOGOUS_ONLY or NOT_FOUND_IN_INDEX with any
                  source failures. Literature could not be queried
                  cleanly, so absence-of-evidence is not evidence.
                  -> accept with medium confidence
    CONTRADICTS   NOT_FOUND_IN_INDEX with all sources healthy.
                  Every source was queried and none returned anything
                  relevant. This is a real signal of novelty/mismatch.
                  -> reject (or require manual review)

The rule above is the *literature* half. The full audit combines it
with symbolic checks:

    accept_discovery(expr) :=
        node_count(expr) <= 8
        AND consistency_error(expr) <= tol
        AND lit_assessment(expr).state != CONTRADICTS

Callers get an AuditResult they can log or route to manual review.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum

from purediscover.litcheck_named import assess_law


class LitState(str, Enum):
    WITNESS = "WITNESS"
    INCONCLUSIVE = "INCONCLUSIVE"
    CONTRADICTS = "CONTRADICTS"


@dataclass
class AuditResult:
    state: LitState
    verdict: str
    confidence: str
    promoted: bool
    strategy: str
    witness_title: str | None
    sources_ok: list[str]
    sources_failed: list[str]
    notes: str = ""

    @property
    def accept_lit(self) -> bool:
        return self.state != LitState.CONTRADICTS


def _classify(a: dict) -> LitState:
    if a["known"]:
        return LitState.WITNESS
    # known == False: verdict is ANALOGOUS_ONLY or NOT_FOUND_IN_INDEX
    if a["verdict"] == "NOT_FOUND_IN_INDEX" and not a["sources_failed"]:
        return LitState.CONTRADICTS
    return LitState.INCONCLUSIVE


def assess_literature(law_name: str, limit_per_source: int = 3,
                      ledger_path: str = "litcheck_ledger.jsonl") -> AuditResult:
    """Return three-state literature assessment for one law."""
    a = assess_law(law_name, limit_per_source=limit_per_source,
                   ledger_path=ledger_path)
    state = _classify(a)
    notes = []
    if state == LitState.INCONCLUSIVE:
        if a["sources_failed"]:
            notes.append(f"sources_failed={a['sources_failed']}")
        if a["verdict"] == "ANALOGOUS_ONLY":
            notes.append("analogous_only")
    if state == LitState.CONTRADICTS:
        notes.append("all_sources_healthy_no_match")
    return AuditResult(
        state=state,
        verdict=a["verdict"],
        confidence=a["confidence"],
        promoted=a["promoted"],
        strategy=a["strategy"],
        witness_title=a["witness_title"],
        sources_ok=a["sources_ok"],
        sources_failed=a["sources_failed"],
        notes="; ".join(notes),
    )


@dataclass
class FullAuditResult:
    accept: bool
    lit: AuditResult
    node_count: int
    consistency_error: float
    max_nodes: int = 8
    tol: float = 0.02
    reasons: list[str] = field(default_factory=list)


def audit_discovery(law_name: str, node_count: int,
                    consistency_error: float,
                    max_nodes: int = 8, tol: float = 0.02,
                    limit_per_source: int = 3,
                    ledger_path: str = "litcheck_ledger.jsonl") -> FullAuditResult:
    """Full audit combining node count, consistency, and literature."""
    reasons = []
    ok = True

    if node_count > max_nodes:
        ok = False
        reasons.append(f"node_count={node_count} > {max_nodes}")

    if consistency_error > tol:
        ok = False
        reasons.append(f"consistency={consistency_error:.4f} > {tol}")

    lit = assess_literature(law_name, limit_per_source=limit_per_source,
                            ledger_path=ledger_path)

    if lit.state == LitState.CONTRADICTS:
        ok = False
        reasons.append("lit_contradicts")

    if ok:
        reasons.append(f"lit_{lit.state.value.lower()}")

    return FullAuditResult(
        accept=ok, lit=lit, node_count=node_count,
        consistency_error=consistency_error,
        max_nodes=max_nodes, tol=tol, reasons=reasons,
    )


__all__ = [
    "LitState", "AuditResult", "FullAuditResult",
    "assess_literature", "audit_discovery",
    "replay_audit",
    "replay_audit_per_law",
]


def replay_audit(ledger_path: str = "litcheck_ledger.jsonl",
                 limit: int | None = None) -> list[dict]:
    """Replay the audit gate over the litcheck ledger.

    Reads ledger_path (append-only JSONL, one Verdict per line), and
    reclassifies each entry under the current 3-state rule:

        WITNESS      verdict in KNOWN_VERDICTS
        CONTRADICTS  NOT_FOUND_IN_INDEX with all sources healthy
        INCONCLUSIVE otherwise

    Returns a list of dicts in file order:
        {fingerprint, expr, verdict, state, accept}

    Missing ledger returns an empty list.
    """
    import json
    from pathlib import Path

    p = Path(ledger_path)
    if not p.exists():
        return []
    lines = [ln for ln in p.read_text().splitlines() if ln.strip()]
    if limit:
        lines = lines[-limit:]

    out = []
    for line in lines:
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        verdict = e.get("verdict", "NOT_FOUND_IN_INDEX")
        failed = e.get("sources_failed", []) or []
        if verdict in KNOWN_VERDICTS:
            state = LitState.WITNESS
        elif verdict == "NOT_FOUND_IN_INDEX" and not failed:
            state = LitState.CONTRADICTS
        else:
            state = LitState.INCONCLUSIVE
        out.append({
            "fingerprint": e.get("fingerprint"),
            "expr": e.get("expr"),
            "verdict": verdict,
            "state": state.value,
            "accept": state is not LitState.CONTRADICTS,
        })
    return out


# --- replay_audit compatibility ---
# KNOWN_VERDICTS lives in litcheck_named.py; alias here so replay_audit
# doesn't need to import the whole wrapper (avoids import cycle risk
# during purediscover/__init__.py loading).
KNOWN_VERDICTS = {"EXACT_IN_INDEX", "STRONG_LEXICAL_MATCH", "FAMILY_IN_INDEX"}

# --- per-law replay with enriched-ledger awareness ---
def _load_enriched(path: str = "purediscover_assess_ledger.jsonl") -> dict[str, dict]:
    """Return {law_name: best_enriched_record}."""
    import json
    from pathlib import Path
    out: dict[str, dict] = {}
    p = Path(path)
    if not p.exists():
        return out
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        name = e.get("name")
        if not name:
            continue
        # prefer any known=true record; else keep first
        if name not in out or (e.get("known") and not out[name].get("known")):
            out[name] = e
    return out


def replay_audit_per_law(ledger_path: str = "litcheck_ledger.jsonl",
                         enriched_path: str = "purediscover_assess_ledger.jsonl",
                         ) -> list[dict]:
    """Return one audit entry per unique law/expr.

    Priority:
      1. enriched ledger record if it exists for that law (has promotion info)
      2. otherwise best per-attempt state from the raw ledger, grouped by expr

    CONTRADICTS is only emitted when the *law* has no WITNESS or
    INCONCLUSIVE attempt anywhere — a single failed attempt against a
    healthy source is not a contradiction, it is a missed query.
    """
    enriched = _load_enriched(enriched_path)

    # group raw ledger by expr
    raw_entries = replay_audit(ledger_path=ledger_path)
    by_expr: dict[str, list[dict]] = {}
    for e in raw_entries:
        by_expr.setdefault(str(e.get("expr")), []).append(e)

    # build per-law output
    out: list[dict] = []
    handled: set[str] = set()

    # enriched first
    for name, e in enriched.items():
        state = (LitState.WITNESS.value if e.get("known")
                 else LitState.INCONCLUSIVE.value)
        out.append({
            "name": name,
            "expr": None,
            "state": state,
            "verdict": e.get("verdict"),
            "promoted": e.get("promoted", False),
            "witness_title": e.get("witness_title"),
            "source": "enriched",
            "accept": True,
        })
        handled.add(name)

    # raw-only laws (grouped by expr)
    for expr, entries in by_expr.items():
        if expr in handled:
            continue
        # best across attempts
        best = max(entries, key=lambda e: _rank_state(e["state"]))
        # reclassify CONTRADICTS: only if NO attempt anywhere was better
        if best["state"] == LitState.CONTRADICTS.value:
            # check if any attempt reached witness/inconclusive
            any_better = any(e["state"] != LitState.CONTRADICTS.value
                             for e in entries)
            if any_better:
                best_state = LitState.INCONCLUSIVE.value
            else:
                best_state = LitState.CONTRADICTS.value
        else:
            best_state = best["state"]
        out.append({
            "name": None,
            "expr": expr,
            "state": best_state,
            "verdict": best["verdict"],
            "promoted": False,
            "witness_title": None,
            "source": "raw",
            "accept": best_state != LitState.CONTRADICTS.value,
        })

    return out


def _rank_state(s: str) -> int:
    return {LitState.WITNESS.value: 3,
            LitState.INCONCLUSIVE.value: 2,
            LitState.CONTRADICTS.value: 1}.get(s, 0)


# --- v2 per-law replay: dedupe raw entries whose expr is in enriched ---
def replay_audit_per_law(ledger_path: str = "litcheck_ledger.jsonl",
                         enriched_path: str = "purediscover_assess_ledger.jsonl",
                         ) -> list[dict]:
    """Return one audit entry per unique law.

    Enriched records (which carry name AND expr) are authoritative for
    any law they cover. Raw entries whose expr appears in any enriched
    record are skipped — they are the same law viewed through the
    per-attempt lens.
    """
    enriched = _load_enriched(enriched_path)
    enriched_exprs: set[str] = set()
    for e in enriched.values():
        x = e.get("expr")
        if x:
            enriched_exprs.add(str(x))

    raw_entries = replay_audit(ledger_path=ledger_path)
    by_expr: dict[str, list[dict]] = {}
    for e in raw_entries:
        by_expr.setdefault(str(e.get("expr")), []).append(e)

    out: list[dict] = []

    # Enriched records first (one per law)
    for name, e in enriched.items():
        state = (LitState.WITNESS.value if e.get("known")
                 else LitState.INCONCLUSIVE.value)
        out.append({
            "name": name,
            "expr": e.get("expr"),
            "state": state,
            "verdict": e.get("verdict"),
            "promoted": e.get("promoted", False),
            "witness_title": e.get("witness_title"),
            "source": "enriched",
            "accept": True,
        })

    # Raw-only laws, deduped against enriched exprs
    for expr, entries in by_expr.items():
        if expr in enriched_exprs:
            continue
        best = max(entries, key=lambda e: _rank_state(e["state"]))
        if best["state"] == LitState.CONTRADICTS.value:
            any_better = any(e["state"] != LitState.CONTRADICTS.value
                             for e in entries)
            best_state = (LitState.INCONCLUSIVE.value if any_better
                          else LitState.CONTRADICTS.value)
        else:
            best_state = best["state"]
        out.append({
            "name": None,
            "expr": expr,
            "state": best_state,
            "verdict": best["verdict"],
            "promoted": False,
            "witness_title": None,
            "source": "raw",
            "accept": best_state != LitState.CONTRADICTS.value,
        })

    return out
