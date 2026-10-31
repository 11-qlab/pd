"""Typed result objects for benchmark runs."""
from __future__ import annotations

import json
import pathlib
from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class LawResult:
    law_name: str
    method: str
    converged: bool
    iterations: int
    points_used: int
    time_s: float
    final_expr: str | None = None
    novelty_class: str | None = None
    skipped: bool = False
    error: str | None = None


@dataclass(frozen=True)
class MethodSummary:
    method: str
    n_total: int
    n_applicable: int
    n_skipped: int
    n_converged: int
    convergence_rate: float
    median_time_s: float | None
    median_iterations: float | None
    median_points: float | None


@dataclass
class BenchmarkResult:
    timestamp: str
    noise_frac: float
    wall_time_s: float
    n_laws: int
    law_names: list[str]
    method_names: list[str]
    results: list[LawResult]
    summaries: list[MethodSummary]
    metadata: dict = field(default_factory=dict)

    def summary_by_method(self) -> dict[str, MethodSummary]:
        return {s.method: s for s in self.summaries}

    def results_for_law(self, law_name: str) -> list[LawResult]:
        return [r for r in self.results if r.law_name == law_name]

    def results_for_method(self, method: str) -> list[LawResult]:
        return [r for r in self.results if r.method == method]

    def to_json(self, path: str | pathlib.Path) -> None:
        payload = {
            "timestamp": self.timestamp,
            "noise_frac": self.noise_frac,
            "wall_time_s": self.wall_time_s,
            "n_laws": self.n_laws,
            "law_names": self.law_names,
            "method_names": self.method_names,
            "results": [asdict(r) for r in self.results],
            "summaries": [asdict(s) for s in self.summaries],
            "metadata": self.metadata,
        }
        pathlib.Path(path).write_text(json.dumps(payload, indent=2))

    @classmethod
    def from_json(cls, path: str | pathlib.Path) -> "BenchmarkResult":
        payload = json.loads(pathlib.Path(path).read_text())
        return cls(
            timestamp=payload["timestamp"],
            noise_frac=payload["noise_frac"],
            wall_time_s=payload["wall_time_s"],
            n_laws=payload["n_laws"],
            law_names=payload["law_names"],
            method_names=payload["method_names"],
            results=[LawResult(**r) for r in payload["results"]],
            summaries=[MethodSummary(**s) for s in payload["summaries"]],
            metadata=payload.get("metadata", {}),
        )


__all__ = ["LawResult", "MethodSummary", "BenchmarkResult"]
