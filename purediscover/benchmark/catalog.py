"""Law catalog: definitions, lookup, data generation."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import sympy as sp


@dataclass(frozen=True)
class Law:
    name: str
    expr: sp.Expr
    var_names: list[str]
    var_ranges: dict[str, tuple[float, float]]
    difficulty: str          # "easy" | "medium" | "hard"
    source: str
    sampling: str = "uniform"
    seed: int = 42


def make_data(law: Law, n: int, noise_frac: float = 0.02,
              seed: int | None = None) -> list[tuple[dict[str, float], float]]:
    """Generate n noisy samples from law."""
    rng = random.Random(seed if seed is not None else law.seed)
    data = []
    for _ in range(n):
        env: dict[str, float] = {}
        for var_name in law.var_names:
            lo, hi = law.var_ranges[var_name]
            if law.sampling == "log" and lo > 0:
                xv = math.exp(rng.uniform(math.log(lo), math.log(hi)))
            else:
                xv = rng.uniform(lo, hi)
            env[var_name] = xv
        try:
            y = float(law.expr.subs({sp.Symbol(k): v for k, v in env.items()}))
        except Exception:
            continue
        if not math.isfinite(y):
            continue
        if noise_frac > 0:
            y *= (1.0 + rng.gauss(0.0, noise_frac))
        data.append((env, y))
    return data


class LawCatalog:
    """Immutable collection of laws."""

    def __init__(self, laws: list[Law]):
        self._laws = tuple(laws)
        self._by_name = {l.name: l for l in laws}

    def __iter__(self):
        return iter(self._laws)

    def __len__(self):
        return len(self._laws)

    def __contains__(self, name: str) -> bool:
        return name in self._by_name

    def get(self, name: str) -> Law:
        if name not in self._by_name:
            raise KeyError(name)
        return self._by_name[name]

    def filter(self, *, difficulty: str | None = None,
               n_vars: int | None = None,
               names: list[str] | None = None) -> "LawCatalog":
        out = list(self._laws)
        if difficulty is not None:
            out = [l for l in out if l.difficulty == difficulty]
        if n_vars is not None:
            out = [l for l in out if len(l.var_names) == n_vars]
        if names is not None:
            wanted = set(names)
            out = [l for l in out if l.name in wanted]
        return LawCatalog(out)

    def names(self) -> list[str]:
        return [l.name for l in self._laws]

    @classmethod
    def default(cls) -> "LawCatalog":
        return cls(_default_laws())


def _default_laws() -> list[Law]:
    A, Ea, T = sp.symbols("A Ea T")
    m, v, a, F = sp.symbols("m v a F")
    I, R = sp.symbols("I R")
    r, x = sp.symbols("r x")
    m1, m2, q1, q2 = sp.symbols("m1 m2 q1 q2")
    t, w, b = sp.symbols("t w b")

    return [
        # ----- EASY (1-var) -----
        Law(name="stefan_boltzmann_T4", expr=T ** 4,
            var_names=["T"], var_ranges={"T": (300.0, 3000.0)},
            difficulty="easy", source="Stefan 1879 / Boltzmann 1884"),
        Law(name="kepler_third", expr=a ** sp.Rational(3, 2),
            var_names=["a"], var_ranges={"a": (0.5, 30.0)},
            difficulty="easy", source="Kepler 1619", sampling="log"),
        Law(name="inverse_square", expr=1 / r ** 2,
            var_names=["r"], var_ranges={"r": (0.5, 10.0)},
            difficulty="easy", source="Bouguer 1729", sampling="log"),
        Law(name="linear", expr=sp.Float(2.0) * x,
            var_names=["x"], var_ranges={"x": (0.5, 10.0)},
            difficulty="easy", source="trivial"),

        # ----- MEDIUM (1-var + simple 2-var) -----
        Law(name="ohm", expr=I * R,
            var_names=["I", "R"],
            var_ranges={"I": (0.1, 5.0), "R": (1.0, 100.0)},
            difficulty="medium", source="Ohm 1827", sampling="log"),
        Law(name="newton_2", expr=m * a,
            var_names=["m", "a"],
            var_ranges={"m": (0.5, 10.0), "a": (1.0, 10.0)},
            difficulty="medium", source="Newton 1687"),
        Law(name="kinetic_energy", expr=m * v ** 2,
            var_names=["m", "v"],
            var_ranges={"m": (0.5, 10.0), "v": (1.0, 20.0)},
            difficulty="medium", source="Coriolis 1829"),
        Law(name="arrhenius", expr=sp.exp(sp.Float(-8000.0) / T),
            var_names=["T"], var_ranges={"T": (280.0, 400.0)},
            difficulty="medium", source="Arrhenius 1889"),
        Law(name="offset_inverse", expr=sp.Float(2.0) / x + sp.Float(1.0),
            var_names=["x"], var_ranges={"x": (0.2, 20.0)},
            difficulty="medium", source="generic (background + 1/x)",
            sampling="log"),

        # ----- HARD (3-var) -----
        Law(name="gravitational", expr=m1 * m2 / r ** 2,
            var_names=["m1", "m2", "r"],
            # Tightened from [1,100] to [1,10] for masses: the 100x range
            # produced a 10^6 y-range that exceeded brute_search's internal
            # tolerance. 10x mass range matches coulomb's 10x charge range
            # and brings the y-range into searchable territory.
            var_ranges={"m1": (1.0, 10.0), "m2": (1.0, 10.0),
                        "r": (1.0, 10.0)},
            difficulty="hard", source="Newton 1687", sampling="log"),
        Law(name="coulomb", expr=q1 * q2 / r ** 2,
            var_names=["q1", "q2", "r"],
            var_ranges={"q1": (0.5, 5.0), "q2": (0.5, 5.0),
                        "r": (0.5, 10.0)},
            difficulty="hard", source="Coulomb 1785", sampling="log"),

        # ----- QUANTUM-GRAVITY-ADJACENT FORMS -----
        # Not settled physics — these are well-defined hypothetical forms
        # that any quantum-gravity candidate must reduce to or extend.
        # The benchmark tests whether the search can recover each form,
        # not whether the form is "the" correct theory.

        Law(name="yukawa_potential",
            # Massive graviton exchange: V(r) = -G m1 m2 * exp(-r/lambda) / r
            # With m1=m2=1 and G folded into a constant, this reduces to
            # exp(-r/lambda)/r, a real 2-variable testable form.
            expr=sp.exp(-r / sp.Symbol("lam")) / r,
            var_names=["r", "lam"],
            var_ranges={"r": (0.5, 10.0), "lam": (0.5, 5.0)},
            difficulty="hard",
            source="Yukawa 1935 (graviton mass analogue)",
            sampling="log"),

        Law(name="newton_1pn",
            # Post-Newtonian 1PN correction to Newtonian gravity:
            # F = G m1 m2 / r^2 * (1 + k/r), with G m1 m2 k folded
            # into a constant. Searchable as 1-variable after collapse.
            expr=sp.Float(1.0) / r ** 2 * (1 + sp.Float(0.5) / r),
            var_names=["r"],
            var_ranges={"r": (0.5, 10.0)},
            difficulty="hard",
            source="Einstein 1915 (post-Newtonian expansion)",
            sampling="log"),

        Law(name="newton_short_distance",
            # Higher-dimensional gravity (ADD/RS) leading correction:
            # F = G m1 m2 / r^2 * (1 + k r^2) at sub-mm scales.
            # Reduces to A/r^2 + A*k, a testable offset-power form.
            expr=(sp.Float(1.0) / r ** 2) * (1 + sp.Float(0.2) * r ** 2),
            var_names=["r"],
            var_ranges={"r": (0.5, 5.0)},
            difficulty="hard",
            source="Arkani-Hamed+ 1998 (large extra dimensions)",
            sampling="log"),
    ]


# Backward-compat shims (old API)
_DEFAULT_CATALOG = LawCatalog.default()
CATALOG: list[Law] = list(_DEFAULT_CATALOG)
def get_law(name: str) -> Law: return _DEFAULT_CATALOG.get(name)
def by_difficulty(level: str) -> list[Law]: return list(_DEFAULT_CATALOG.filter(difficulty=level))


__all__ = ["Law", "LawCatalog", "make_data", "CATALOG", "get_law", "by_difficulty"]
