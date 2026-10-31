"""Discovery stratum with literature gate."""
from __future__ import annotations
import sympy as sp
from purediscover import Grammar, discover_law, replay_audit
from purediscover.litcheck import litcheck, Verdict


PROMOTABLE = {"NOT_FOUND_IN_INDEX", "ANALOGOUS_ONLY"}
MAX_NODE_COUNT = 8          # anything above this is curve fitting
MIN_AUDIT_CONF = "low"


class DiscoverVerifier:
    def check(self, r):
        out = r.output

        # 1. MDL gate (already applied by solver, re-check)
        if out.get("bits") is None:
            return False

        # 2. audit gate
        audit = replay_audit(out["expr"], out["held_out"],
                             tolerance=out.get("tol", 0.05))
        if not audit["ok"]:
            return False

        # 3. node-count gate — high node count is not a law
        n = sp.count_ops(out["expr"]) + len(out["expr"].free_symbols)
        if n > MAX_NODE_COUNT:
            return False

        # 4. literature gate
        v: Verdict = litcheck(
            out["expr"],
            var_descriptions=out.get("var_descriptions"),
            ledger_path=out.get("ledger", "litcheck_ledger.jsonl"),
        )
        out["litcheck"] = {
            "verdict": v.verdict,
            "confidence": v.confidence,
            "top_doi": [p.doi for p in v.top_papers[:3] if p.doi],
        }
        return v.verdict in PROMOTABLE


class DiscoverService:
    def solve(self, task):
        spec = task.spec
        g = Grammar(**spec["grammar"])
        cands = discover_law(
            g, spec["data"],
            n_candidates=spec.get("n", 6000),
            seed=spec.get("seed", 0),
            tol=spec.get("tol", 0.05),
            beam=spec.get("beam", 64),
            time_budget=spec.get("budget", 120.0),
        )
        if not cands:
            return None
        best = cands[0]
        return {
            "expr": best["expr"],
            "bits": best["score"]["bits"],
            "held_out": spec["held_out"],
            "tol": spec.get("tol", 0.05),
            "var_descriptions": spec.get("var_descriptions"),
        }


def init(net):
    net.register_service("symbolic_discover", DiscoverService())
    net.register_stratum("discover", DiscoverVerifier(), [])
    net.run_forever(stratum="discover")
