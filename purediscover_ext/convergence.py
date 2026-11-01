"""Convergence monitor with best-value-tracking logic."""
from __future__ import annotations

import re
from dataclasses import dataclass, field


class LoopStopped(Exception):
    pass


def _fingerprint(expr_str: str) -> str:
    s = str(expr_str)
    s = re.sub(r"\d+\.\d+e?-?\d*", "", s)
    s = re.sub(r"\d+\.\d+", "", s)
    s = re.sub(r"(?<![A-Za-z])\d+(?![A-Za-z])", "", s)
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"[A-Za-z_][A-Za-z_0-9]*",
               lambda m: m.group(0) if m.group(0) in
               ("cos", "sin", "tan", "exp", "log", "sqrt", "abs", "tanh")
               else "V", s)
    return s


@dataclass
class ConvergenceMonitor:
    patience: int = 5
    mae_epsilon: float = 5e-3
    bits_epsilon: float = 2.0
    history: list = field(default_factory=list)

    def update(self, best_candidate: dict | None) -> dict:
        if best_candidate is None:
            self.history.append({"expr": None, "bits": float("inf"),
                                 "mae": float("inf"), "fp": None})
            return {"converged": False, "reason": "no_candidate", "streak": 0}

        bits = float(best_candidate.get("bits", best_candidate.get("mdl_bits",
                    float("inf"))))
        mae  = float(best_candidate.get("mae",  best_candidate.get("mre",
                    best_candidate.get("fit_error", float("inf")))))
        expr = str(best_candidate.get("expr", best_candidate.get("expr_str", "")))
        fp   = _fingerprint(expr)
        self.history.append({"expr": expr, "bits": bits, "mae": mae, "fp": fp})

        n = len(self.history)
        if n < self.patience + 1:
            return {"converged": False, "reason": "insufficient_history",
                    "streak": 0}

        recent = self.history[-(self.patience + 1):]
        earlier = self.history[:-1]

        recent_best_mae  = min(r["mae"]  for r in recent)
        earlier_best_mae = min(r["mae"]  for r in earlier)
        recent_best_bits  = min(r["bits"] for r in recent)
        earlier_best_bits = min(r["bits"] for r in earlier)

        mae_improving  = (earlier_best_mae  - recent_best_mae)  > self.mae_epsilon
        bits_improving = (earlier_best_bits - recent_best_bits) > self.bits_epsilon

        mae_plateau  = not mae_improving
        bits_plateau = not bits_improving

        recent_fps = [r["fp"] for r in self.history[-self.patience:]]
        fp_stable = all(f == recent_fps[0] for f in recent_fps) \
                    and recent_fps[0] is not None

        streak = 0
        for r in reversed(self.history):
            if r["fp"] == fp:
                streak += 1
            else:
                break

        converged = bool(mae_plateau and bits_plateau and fp_stable
                         and streak >= self.patience)

        reason = "converged" if converged else (
            "mae_improving" if mae_improving else
            "bits_improving" if bits_improving else
            "structure_changed"
        )

        return {
            "converged": converged, "reason": reason, "streak": streak,
            "mae": mae, "bits": bits, "fp": fp,
            "recent_best_mae": recent_best_mae,
            "recent_best_bits": recent_best_bits,
        }

    def summary(self) -> dict:
        if not self.history:
            return {"n_iterations": 0}
        valid = [r for r in self.history
                 if r["mae"] != float("inf") and r["bits"] != float("inf")]
        if not valid:
            return {"n_iterations": len(self.history)}
        # Pick by bits; among candidates within 0.5% of best bits, prefer
        # the latest iteration (most data → most refined coefficients).
        best_bits = min(r["bits"] for r in valid)
        near_best = [r for r in valid if r["bits"] <= best_bits * 1.005]
        best = near_best[-1]  # latest among near-best
        return {
            "n_iterations": len(self.history),
            "best_bits": best["bits"],
            "best_mae":  best["mae"],
            "best_expr": best["expr"],
            "final_expr": self.history[-1]["expr"],
        }
