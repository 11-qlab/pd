"""Novelty gate — five-layer filter for candidate discoveries."""
from purediscover.novelty_gate.statistical import statistical_gate
from purediscover.novelty_gate.dimensional import dimensional_check, DIMENSIONS
from purediscover.novelty_gate.literature import literature_gate, algebraic_fingerprint
from purediscover.novelty_gate.gate import (
    full_gate, render_report, replication_gate,
)

__all__ = [
    "statistical_gate", "dimensional_check", "DIMENSIONS",
    "literature_gate", "algebraic_fingerprint",
    "full_gate", "render_report", "replication_gate",
]
