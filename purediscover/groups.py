"""Small finite group exploration. No GAP.

Sympy named-group constructors for the main classes. Q8 has no sympy
constructor and is built by its regular permutation representation on 8
points (left multiplication in the quaternion group)."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
from sympy.combinatorics import Permutation, PermutationGroup
from sympy.combinatorics.named_groups import (
    CyclicGroup, DihedralGroup, SymmetricGroup,
    AlternatingGroup, AbelianGroup,
)


def _q8() -> PermutationGroup:
    # Left-regular rep of Q8 = <i, j | i^4=1, i^2=j^2, j i j^-1 = i^-1>
    # Label elements 0..7 = 1, i, -1, -i, j, k, -j, -k
    # i acts as (0 1 2 3)(4 5 6 7)
    # j acts as (0 4 2 6)(1 7 3 5)
    i = Permutation(0, 1, 2, 3)(4, 5, 6, 7)
    j = Permutation(0, 4, 2, 6)(1, 7, 3, 5)
    return PermutationGroup([i, j])


@dataclass
class GroupCandidate:
    name: str
    order: int
    abelian: bool
    description: str
    group: object = None
    score: float = 0.0


_REGISTRY = [
    ("Z2",    2,  True,  "cyclic 2",       lambda: CyclicGroup(2)),
    ("Z3",    3,  True,  "cyclic 3",       lambda: CyclicGroup(3)),
    ("Z4",    4,  True,  "cyclic 4",       lambda: CyclicGroup(4)),
    ("V4",    4,  True,  "klein four",     lambda: AbelianGroup(2, 2)),
    ("Z5",    5,  True,  "cyclic 5",       lambda: CyclicGroup(5)),
    ("Z6",    6,  True,  "cyclic 6",       lambda: CyclicGroup(6)),
    ("S3",    6,  False, "symmetric 3",    lambda: SymmetricGroup(3)),
    ("Z7",    7,  True,  "cyclic 7",       lambda: CyclicGroup(7)),
    ("Z8",    8,  True,  "cyclic 8",       lambda: CyclicGroup(8)),
    ("Z4xZ2", 8,  True,  "Z4 x Z2",        lambda: AbelianGroup(4, 2)),
    ("Z2^3",  8,  True,  "Z2^3",           lambda: AbelianGroup(2, 2, 2)),
    ("D4",    8,  False, "dihedral 4",     lambda: DihedralGroup(4)),
    ("Q8",    8,  False, "quaternion 8",   _q8),
    ("Z9",    9,  True,  "cyclic 9",       lambda: CyclicGroup(9)),
    ("Z3^2",  9,  True,  "Z3^2",           lambda: AbelianGroup(3, 3)),
    ("Z10",  10,  True,  "cyclic 10",      lambda: CyclicGroup(10)),
    ("D5",   10,  False, "dihedral 5",     lambda: DihedralGroup(5)),
    ("Z11",  11,  True,  "cyclic 11",      lambda: CyclicGroup(11)),
    ("Z12",  12,  True,  "cyclic 12",      lambda: CyclicGroup(12)),
    ("A4",   12,  False, "alternating 4",  lambda: AlternatingGroup(4)),
    ("D6",   12,  False, "dihedral 6",     lambda: DihedralGroup(6)),
    ("Z13",  13,  True,  "cyclic 13",      lambda: CyclicGroup(13)),
    ("Z14",  14,  True,  "cyclic 14",      lambda: CyclicGroup(14)),
    ("D7",   14,  False, "dihedral 7",     lambda: DihedralGroup(7)),
    ("Z15",  15,  True,  "cyclic 15",      lambda: CyclicGroup(15)),
]


def _all_small_groups(max_order: int = 15) -> list[GroupCandidate]:
    out = []
    for name, order, ab, desc, factory in _REGISTRY:
        if order <= max_order:
            try:
                g = factory()
            except Exception:
                g = None
            out.append(GroupCandidate(name, order, ab, desc, group=g))
    return out


def explore_groups(
    score_fn: Callable[[GroupCandidate], float],
    max_order: int = 12,
    top_k: int = 5,
) -> list[GroupCandidate]:
    cands = _all_small_groups(max_order)
    for c in cands:
        try:
            c.score = float(score_fn(c))
        except Exception:
            c.score = float("-inf")
    return sorted(cands, key=lambda c: c.score, reverse=True)[:top_k]
