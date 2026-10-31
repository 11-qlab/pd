"""Known-law resource corpus.

Every entry is a Law: sympy expression, variable descriptions, units,
domain, source, difficulty, ops needed. Used for:
  - benchmark: can discovery rediscover it?
  - validation: does litcheck return known?
  - curriculum: sort by difficulty, feed as tasks
"""
from purediscover.resource.laws import Law, REGISTRY, by_domain, by_difficulty, get
from purediscover.resource.constants import CONSTANTS, get_constant
from purediscover.resource.notation import SYNONYMS, normalize_var

__all__ = [
    "Law", "REGISTRY", "by_domain", "by_difficulty", "get",
    "CONSTANTS", "get_constant", "SYNONYMS", "normalize_var",
]
