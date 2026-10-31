"""Abstract base class for discovery methods."""
from __future__ import annotations

from abc import ABC, abstractmethod

from purediscover.benchmark.catalog import Law
from purediscover.benchmark.results import LawResult


class DiscoveryMethod(ABC):
    name: str = "base"

    @abstractmethod
    def run(self, law: Law, **kwargs) -> LawResult:
        """Execute this method on a single law."""

    def supports(self, law: Law) -> bool:
        """Can this method attempt the given law?"""
        return True
