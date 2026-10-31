"""Discovery stratum — plug-in for the eternity net."""
from __future__ import annotations
from purediscover import Grammar, discover_law, replay_audit


class DiscoverVerifier:
    """A discovery is valid iff it replays on held-out data."""

    def check(self, r):
        out = r.output
        return replay_audit(
            out["expr"], out["held_out"],
            tolerance=out.get("tol", 0.05),
        )["ok"]


class DiscoverService:
    """Node-side solver. Deterministic given seed."""

    def solve(self, task):
        spec = task.spec
        g = Grammar(**spec["grammar"])
        cands = discover_law(
            g, spec["data"],
            n_candidates=spec.get("n", 5000),
            seed=spec.get("seed", 0),
            tol=spec.get("tol", 0.05),
            beam=spec.get("beam", 64),
            time_budget=spec.get("budget", 60.0),
        )
        if not cands:
            return None
        return {
            "expr": cands[0]["expr"],
            "bits": cands[0]["score"]["bits"],
            "held_out": spec["held_out"],
            "tol": spec.get("tol", 0.05),
        }


def init(net):
    net.register_service("symbolic_discover", DiscoverService())
    net.register_stratum("discover", DiscoverVerifier(), [])
    net.run_forever(stratum="discover")
