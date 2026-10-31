"""Interferometric phase laws — algebraicized forms.

SCOPE: Interferometric phases are DERIVED consequences of quantum
mechanics plus an external potential. They are not independent laws of
nature. The pipeline's job is to recover the algebraic relationship
between experimental parameters and measured phase — the same task it
performs for fundamental laws. We include these forms because:

    (1) They are experimentally measurable (COW 1975 was the first;
        many since, including gravitational AB, Compton clock phase).
    (2) They test the pipeline on physics-adjacent forms that are NOT
        the underlying fundamental laws.
    (3) They demonstrate coverage of contemporary precision-measurement
        observables beyond classical mechanics.

PAPER CAVEAT: These are NOT fundamental laws. If the paper says
"recovers 9 interferometric phase laws," that's fine. If it says
"discovers 9 new laws of quantum mechanics," that's false. Phrase
carefully: "recovers the algebraic forms that current interferometric
experiments measure."

Sources:
    COW        Colella, Overhauser, Werner, PRL 34, 1472 (1975)
    Sagnac     Sagnac, Comptes Rendus 157, 708 (1913)
    AB         Aharonov, Bohm, Phys. Rev. 115, 485 (1959)
    AC         Aharonov, Casher, PRL 53, 319 (1984)
    GravAB     Anandan, PRL 52, 2692 (1984);
               Hohensee et al., PRL 106, 151102 (2011)
    Compton    Zych et al., Nature Physics 7, 235 (2011);
               Sinha, Samuel, PRL 106, 140401 (2011)
"""
from __future__ import annotations

import sympy as sp

from purediscover.benchmark.catalog import Law, LawCatalog


def build_phase_laws() -> list[Law]:
    m, g, T, Omega, A, q, B, V, t, omega, h, L, dh, mu, E = sp.symbols(
        "m g T Omega A q B V t omega h L dh mu E"
    )

    _r = (0.5, 5.0)

    def _ranges(*names):
        return {n: _r for n in names}

    return [
        Law(
            name="cow_phase",
            expr=m * g * T ** 2,
            var_names=["m", "g", "T"],
            var_ranges=_ranges("m", "g", "T"),
            difficulty="medium",
            source="Colella-Overhauser-Werner 1975",
            sampling="log",
        ),
        Law(
            name="sagnac_phase",
            expr=m * Omega * A,
            var_names=["m", "Omega", "A"],
            var_ranges=_ranges("m", "Omega", "A"),
            difficulty="medium",
            source="Sagnac 1913 (matter wave)",
            sampling="log",
        ),
        Law(
            name="magnetic_ab",
            expr=q * B * A,
            var_names=["q", "B", "A"],
            var_ranges=_ranges("q", "B", "A"),
            difficulty="medium",
            source="Aharonov-Bohm 1959",
            sampling="log",
        ),
        Law(
            name="electric_ab",
            expr=q * V * t,
            var_names=["q", "V", "t"],
            var_ranges=_ranges("q", "V", "t"),
            difficulty="medium",
            source="Aharonov-Bohm 1959",
            sampling="log",
        ),
        Law(
            name="grav_ab_photon",
            expr=omega * g * h * L,
            var_names=["omega", "g", "h", "L"],
            var_ranges=_ranges("omega", "g", "h", "L"),
            difficulty="hard",
            source="Anandan 1984; Hohensee et al. 2011",
            sampling="log",
        ),
        Law(
            name="compton_clock",
            expr=m * g * dh * T,
            var_names=["m", "g", "dh", "T"],
            var_ranges=_ranges("m", "g", "dh", "T"),
            difficulty="hard",
            source="Zych et al. 2011; Sinha & Samuel 2011",
            sampling="log",
        ),
        Law(
            name="aharonov_casher",
            expr=mu * E * L,
            var_names=["mu", "E", "L"],
            var_ranges=_ranges("mu", "E", "L"),
            difficulty="medium",
            source="Aharonov-Casher 1984",
            sampling="log",
        ),
        Law(
            name="scalar_ab",
            expr=m * V * t,
            var_names=["m", "V", "t"],
            var_ranges=_ranges("m", "V", "t"),
            difficulty="medium",
            source="scalar Aharonov-Bohm",
            sampling="log",
        ),
        Law(
            name="atom_grav_gradient",
            expr=m * g * T ** 2 + sp.Float(0.1) * g * T ** 4,
            var_names=["m", "g", "T"],
            var_ranges=_ranges("m", "g", "T"),
            difficulty="hard",
            source="atom interferometry gravity gradient (leading order)",
            sampling="log",
        ),
    ]


class PhaseCatalog(LawCatalog):
    @classmethod
    def default(cls) -> "PhaseCatalog":
        return cls(build_phase_laws())


def full_catalog() -> LawCatalog:
    """Classical + interferometric combined."""
    return LawCatalog(list(LawCatalog.default()) + list(PhaseCatalog.default()))


__all__ = ["PhaseCatalog", "build_phase_laws", "full_catalog"]
