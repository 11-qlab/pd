"""Physical constants in SI. Used to un-strip laws when reconstructing
full forms or when generating data that needs a real scale.
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class Constant:
    name: str
    symbol: str
    value: float
    unit: str
    uncertainty: float = 0.0


CONSTANTS: dict[str, Constant] = {
    "c":      Constant("speed of light",       "c",   299_792_458.0,  "m/s", 0.0),
    "G":      Constant("gravitational const",  "G",   6.67430e-11,    "m³/(kg·s²)", 1.5e-15),
    "h":      Constant("Planck constant",      "h",   6.62607015e-34,"J·s", 0.0),
    "hbar":   Constant("reduced Planck",       "ħ",   1.054571817e-34,"J·s", 0.0),
    "k_B":    Constant("Boltzmann constant",   "k_B", 1.380649e-23,  "J/K", 0.0),
    "e":      Constant("elementary charge",    "e",   1.602176634e-19,"C", 0.0),
    "N_A":    Constant("Avogadro number",      "N_A", 6.02214076e23, "1/mol", 0.0),
    "R":      Constant("gas constant",         "R",   8.314462618,   "J/(mol·K)", 0.0),
    "g":      Constant("standard gravity",     "g",   9.80665,       "m/s²", 0.0),
    "mu_0":   Constant("vacuum permeability",  "μ₀",  1.25663706212e-6,"N/A²", 0.0),
    "eps_0":  Constant("vacuum permittivity",  "ε₀",  8.8541878128e-12,"F/m", 0.0),
    "sigma":  Constant("Stefan-Boltzmann",     "σ",   5.670374419e-8,"W/(m²·K⁴)", 0.0),
    "G_F":    Constant("Fermi constant",       "G_F", 1.1663787e-5,  "GeV⁻²", 0.0),
    "alpha":  Constant("fine structure",       "α",   7.2973525693e-3,"1", 0.0),
    "m_e":    Constant("electron mass",        "m_e", 9.1093837015e-31,"kg", 0.0),
    "m_p":    Constant("proton mass",          "m_p", 1.67262192369e-27,"kg", 0.0),
}


def get_constant(symbol: str) -> Constant:
    if symbol in CONSTANTS:
        return CONSTANTS[symbol]
    for c in CONSTANTS.values():
        if c.symbol == symbol:
            return c
    raise KeyError(symbol)


def unstrip(expr, mapping: dict[str, str]):
    """Substitute constants back into a stripped expression.

    Example: unstrip(a*m, {"__scale__": "1"}) -> a*m
    For kinetic energy stripped to m*v**2, unstrip with
    {"__scale__": "Rational(1,2)"} -> m*v**2/2
    """
    import sympy as sp
    subs = {sp.Symbol(k): sp.sympify(v) for k, v in mapping.items()}
    return expr.subs(subs)
