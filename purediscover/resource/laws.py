"""Known-law registry. Stripped forms: unit constants removed so the
discovery engine can find shape without needing G, h, k, c, ½, π.

`expr` is the sympy expression discovery must reach (up to scalar).
`full_form` is the conventional notation for human readers.
`variables` maps each symbol to a natural-language description used by
litcheck. `ops_needed` is the minimum grammar ops for rediscovery.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import sympy as sp


@dataclass(frozen=True)
class Law:
    name: str
    domain: str
    expr: sp.Expr
    full_form: str
    variables: dict[str, str]
    units: str
    source: str
    difficulty: int
    ops_needed: tuple[str, ...]
    notes: str = ""


def _L(*args, **kwargs) -> Law:
    return Law(*args, **kwargs)


REGISTRY: list[Law] = [
    # ---------- mechanics ----------
    _L("newton_2", "mechanics", sp.Symbol("m") * sp.Symbol("a"),
       "F = m·a", {"m": "mass", "a": "acceleration"},
       "N = kg·m/s²", "Newton, Principia (1687)", 3, ("mul",)),
    _L("kinetic_energy", "mechanics", sp.Symbol("m") * sp.Symbol("v") ** 2,
       "E = ½·m·v²", {"m": "mass", "v": "velocity"},
       "J = kg·(m/s)²", "Coriolis (1829)", 5, ("mul", "pow")),
    _L("gravitational_force", "mechanics",
       sp.Symbol("m1") * sp.Symbol("m2") / sp.Symbol("r") ** 2,
       "F = G·m₁·m₂/r²", {"m1": "mass one", "m2": "mass two", "r": "separation distance"},
       "N", "Newton (1687)", 7, ("mul", "div", "pow")),
    _L("momentum", "mechanics", sp.Symbol("m") * sp.Symbol("v"),
       "p = m·v", {"m": "mass", "v": "velocity"},
       "kg·m/s", "Newton (1687)", 3, ("mul",)),
    _L("angular_momentum", "mechanics",
       sp.Symbol("m") * sp.Symbol("v") * sp.Symbol("r"),
       "L = m·v·r", {"m": "mass", "v": "velocity", "r": "radius"},
       "kg·m²/s", "Euler (1744)", 5, ("mul",)),
    _L("centripetal_accel", "mechanics",
       sp.Symbol("v") ** 2 / sp.Symbol("r"),
       "a = v²/r", {"v": "velocity", "r": "radius"},
       "m/s²", "Huygens (1673)", 5, ("mul", "div", "pow")),
    _L("hooke_law", "mechanics", sp.Symbol("k") * sp.Symbol("x"),
       "F = −k·x", {"k": "spring constant", "x": "displacement"},
       "N", "Hooke (1676)", 3, ("mul",)),
    _L("pendulum_period", "mechanics", sp.sqrt(sp.Symbol("L")),
       "T = 2π·√(L/g)", {"L": "length"},
       "s", "Huygens (1673)", 2, ("sqrt",),
       "constant 2π/√g absorbed by scalar rescale"),
    _L("torque", "mechanics", sp.Symbol("r") * sp.Symbol("F"),
       "τ = r·F", {"r": "lever arm", "F": "force"},
       "N·m", "Euler (1775)", 3, ("mul",)),
    _L("mech_power", "mechanics", sp.Symbol("F") * sp.Symbol("v"),
       "P = F·v", {"F": "force", "v": "velocity"},
       "W", "Watt (1782)", 3, ("mul",)),
    _L("work", "mechanics", sp.Symbol("F") * sp.Symbol("d"),
       "W = F·d", {"F": "force", "d": "displacement"},
       "J", "Coriolis (1829)", 3, ("mul",)),
    _L("distance_kinematic", "mechanics", sp.Symbol("v") * sp.Symbol("t"),
       "d = v·t", {"v": "velocity", "t": "time"},
       "m", "Galileo (1638)", 3, ("mul",)),

    # ---------- electromagnetism ----------
    _L("coulomb", "electromagnetism",
       sp.Symbol("q1") * sp.Symbol("q2") / sp.Symbol("r") ** 2,
       "F = k·q₁·q₂/r²", {"q1": "charge one", "q2": "charge two", "r": "separation distance"},
       "N", "Coulomb (1785)", 7, ("mul", "div", "pow")),
    _L("ohm", "electromagnetism", sp.Symbol("I") * sp.Symbol("R"),
       "V = I·R", {"I": "current", "R": "resistance"},
       "V", "Ohm (1827)", 3, ("mul",)),
    _L("electric_power", "electromagnetism", sp.Symbol("I") * sp.Symbol("V"),
       "P = I·V", {"I": "current", "V": "voltage"},
       "W", "Joule (1840)", 3, ("mul",)),
    _L("lorentz_1d", "electromagnetism", sp.Symbol("q") * sp.Symbol("E"),
       "F = q·E", {"q": "charge", "E": "electric field"},
       "N", "Lorentz (1892)", 3, ("mul",)),
    _L("lorentz_2d", "electromagnetism",
       sp.Symbol("q") * sp.Symbol("v") * sp.Symbol("B"),
       "F = q·v·B", {"q": "charge", "v": "velocity", "B": "magnetic field"},
       "N", "Lorentz (1892)", 5, ("mul",)),
    _L("biot_savart", "electromagnetism",
       sp.Symbol("I") / sp.Symbol("r"),
       "B = μ₀·I/(2π·r)", {"I": "current", "r": "distance"},
       "T", "Biot & Savart (1820)", 3, ("div",)),
    _L("capacitor_energy", "electromagnetism",
       sp.Symbol("C") * sp.Symbol("V") ** 2,
       "E = ½·C·V²", {"C": "capacitance", "V": "voltage"},
       "J", "Volta / Kelvin", 5, ("mul", "pow")),
    _L("inductor_energy", "electromagnetism",
       sp.Symbol("L") * sp.Symbol("I") ** 2,
       "E = ½·L·I²", {"L": "inductance", "I": "current"},
       "J", "Maxwell (1865)", 5, ("mul", "pow")),

    # ---------- thermodynamics ----------
    _L("ideal_gas", "thermodynamics", sp.Symbol("n") * sp.Symbol("T"),
       "P·V = n·R·T", {"n": "amount", "T": "temperature"},
       "J", "Clapeyron (1834)", 3, ("mul",),
       "stripped: T = n·R/P; find n·T"),
    _L("heat_capacity", "thermodynamics", sp.Symbol("m") * sp.Symbol("T"),
       "Q = m·c·ΔT", {"m": "mass", "T": "temperature change"},
       "J", "Black (1760)", 3, ("mul",),
       "c absorbed by scalar rescale"),
    _L("carnot_eff", "thermodynamics",
       1 - sp.Symbol("Tc") / sp.Symbol("Th"),
       "η = 1 − T_c/T_h", {"Tc": "cold reservoir temperature", "Th": "hot reservoir temperature"},
       "dimensionless", "Carnot (1824)", 6, ("sub", "div")),
    _L("stefan_boltzmann", "thermodynamics", sp.Symbol("T") ** 4,
       "j = σ·T⁴", {"T": "temperature"},
       "W/m²", "Stefan (1879) / Boltzmann (1884)", 2, ("pow",),
       "T⁴ requires pow with exponent 4, not just 2,3"),
    _L("boltzmann_entropy", "thermodynamics", sp.log(sp.Symbol("W")),
       "S = k·ln(W)", {"W": "microstate count"},
       "J/K", "Boltzmann (1877)", 2, ("log",)),

    # ---------- quantum ----------
    _L("photon_energy", "quantum", sp.Symbol("f"),
       "E = h·f", {"f": "frequency"},
       "J", "Planck (1900)", 1, (),
       "stripped of h; trivially linear in f"),
    _L("de_broglie", "quantum", 1 / sp.Symbol("p"),
       "λ = h/p", {"p": "momentum"},
       "m", "de Broglie (1924)", 2, ("div",)),
    _L("hydrogen_levels", "quantum", 1 / sp.Symbol("n") ** 2,
       "E_n = −13.6/n² eV", {"n": "principal quantum number"},
       "eV", "Bohr (1913)", 4, ("div", "pow")),
    _L("bohr_radius", "quantum", sp.Symbol("n") ** 2,
       "r_n = a₀·n²", {"n": "principal quantum number"},
       "m", "Bohr (1913)", 2, ("pow",)),

    # ---------- relativity ----------
    _L("emc2", "relativity", sp.Symbol("m"),
       "E = m·c²", {"m": "mass"},
       "J", "Einstein (1905)", 1, (),
       "stripped of c²; linear in m"),
    _L("energy_momentum", "relativity",
       sp.Symbol("p") ** 2 + sp.Symbol("m") ** 2,
       "E² = (p·c)² + (m·c²)²", {"p": "momentum", "m": "mass"},
       "J²", "Einstein (1905)", 7, ("add", "pow")),

    # ---------- optics ----------
    _L("magnification", "optics",
       sp.Symbol("di") / sp.Symbol("do"),
       "m = d_i/d_o", {"di": "image distance", "do": "object distance"},
       "dimensionless", "Snell / Kepler", 3, ("div",)),
    _L("inverse_square", "optics", 1 / sp.Symbol("r") ** 2,
       "I = 1/r²", {"r": "distance"},
       "1/m²", "Bouguer (1729)", 4, ("div", "pow")),

    # ---------- waves ----------
    _L("wave_speed", "waves", sp.Symbol("lam") * sp.Symbol("f"),
       "v = λ·f", {"lam": "wavelength", "f": "frequency"},
       "m/s", "d'Alembert (1747)", 3, ("mul",)),
    _L("period_freq", "waves", 1 / sp.Symbol("f"),
       "T = 1/f", {"f": "frequency"},
       "s", "Euler (1748)", 2, ("div",)),

    # ---------- fluids ----------
    _L("flow_rate", "fluids", sp.Symbol("A") * sp.Symbol("v"),
       "Q = A·v", {"A": "cross-sectional area", "v": "velocity"},
       "m³/s", "Castelli (1628)", 3, ("mul",)),
    _L("pressure", "fluids", sp.Symbol("F") / sp.Symbol("A"),
       "P = F/A", {"F": "force", "A": "area"},
       "Pa", "Pascal (1653)", 3, ("div",)),
    _L("hydrostatic", "fluids",
       sp.Symbol("rho") * sp.Symbol("h"),
       "P = ρ·g·h", {"rho": "density", "h": "depth"},
       "Pa", "Stevin (1586)", 3, ("mul",),
       "g absorbed"),
    _L("reynolds", "fluids",
       sp.Symbol("rho") * sp.Symbol("v") * sp.Symbol("D") / sp.Symbol("mu"),
       "Re = ρ·v·D/μ", {"rho": "density", "v": "velocity", "D": "diameter", "mu": "viscosity"},
       "dimensionless", "Reynolds (1883)", 7, ("mul", "div")),

    # ---------- chemistry ----------
    _L("rate_law_bimol", "chemistry",
       sp.Symbol("A") * sp.Symbol("B"),
       "rate = k·[A]·[B]", {"A": "concentration A", "B": "concentration B"},
       "M/s", "Guldberg & Waage (1864)", 3, ("mul",),
       "k absorbed"),
    _L("ph", "chemistry", -sp.log(sp.Symbol("H")),
       "pH = −log[H⁺]", {"H": "hydrogen ion concentration"},
       "dimensionless", "Sørensen (1909)", 2, ("log",),
       "negative sign absorbed into scalar"),

    # ---------- biology ----------
    _L("michaelis_menten", "biology",
       sp.Symbol("S") / (sp.Symbol("K") + sp.Symbol("S")),
       "v = V·S/(K_m + S)", {"S": "substrate concentration", "K": "Michaelis constant"},
       "M/s", "Michaelis & Menten (1913)", 5, ("div", "add")),
    _L("logistic_growth", "biology",
       sp.Symbol("N") * (1 - sp.Symbol("N") / sp.Symbol("K")),
       "dN/dt = r·N·(1 − N/K)", {"N": "population", "K": "carrying capacity"},
       "1/time", "Verhulst (1838)", 8, ("add", "sub", "mul", "div"),
       "r absorbed"),

    # ---------- mathematics ----------
    _L("area_rect", "mathematics", sp.Symbol("w") * sp.Symbol("h"),
       "A = w·h", {"w": "width", "h": "height"},
       "m²", "Euclid", 3, ("mul",)),
    _L("area_circle", "mathematics", sp.Symbol("r") ** 2,
       "A = π·r²", {"r": "radius"},
       "m²", "Archimedes", 2, ("pow",)),
    _L("volume_sphere", "mathematics", sp.Symbol("r") ** 3,
       "V = (4/3)·π·r³", {"r": "radius"},
       "m³", "Archimedes", 2, ("pow3",),
       "pow³ — needs integer exponent 3"),
    _L("pythagoras_sq", "mathematics",
       sp.Symbol("a") ** 2 + sp.Symbol("b") ** 2,
       "c² = a² + b²", {"a": "leg one", "b": "leg two"},
       "m²", "Pythagoras", 5, ("add", "pow")),
    _L("hypotenuse", "mathematics",
       sp.sqrt(sp.Symbol("a") ** 2 + sp.Symbol("b") ** 2),
       "c = √(a² + b²)", {"a": "leg one", "b": "leg two"},
       "m", "Pythagoras", 6, ("add", "pow", "sqrt")),
    _L("triangle_area", "mathematics",
       sp.Symbol("b") * sp.Symbol("h") / 2,
       "A = ½·b·h", {"b": "base", "h": "height"},
       "m²", "Euclid", 5, ("mul", "div")),

    # ---------- probability ----------
    _L("bayes", "probability",
       sp.Symbol("PB") * sp.Symbol("PA_given_B") / sp.Symbol("PA"),
       "P(A|B) = P(B|A)·P(A)/P(B)",
       {"PB": "probability of B", "PA_given_B": "probability of A given B", "PA": "probability of A"},
       "dimensionless", "Bayes (1763)", 7, ("mul", "div")),
    _L("variance", "probability",
       sp.Symbol("Ex2") - sp.Symbol("Ex") ** 2,
       "Var = E[x²] − (E[x])²", {"Ex2": "expected x squared", "Ex": "expected x"},
       "unit²", "Pearson (1895)", 7, ("sub", "pow")),
    _L("covariance", "probability",
       sp.Symbol("Exy") - sp.Symbol("Ex") * sp.Symbol("Ey"),
       "Cov = E[xy] − E[x]·E[y]",
       {"Exy": "expected x y", "Ex": "expected x", "Ey": "expected y"},
       "unit²", "Pearson (1896)", 7, ("sub", "mul")),

    # ---------- engineering / other ----------
    _L("rect_perimeter", "engineering",
       2 * (sp.Symbol("a") + sp.Symbol("b")),
       "P = 2(a + b)", {"a": "side one", "b": "side two"},
       "m", "Euclid", 5, ("add", "mul")),
    _L("cylinder_volume", "engineering",
       sp.Symbol("r") ** 2 * sp.Symbol("h"),
       "V = π·r²·h", {"r": "radius", "h": "height"},
       "m³", "Archimedes", 5, ("mul", "pow")),
    _L("dilution", "engineering",
       sp.Symbol("C1") * sp.Symbol("V1") / sp.Symbol("V2"),
       "C₂ = C₁·V₁/V₂", {"C1": "initial concentration", "V1": "initial volume", "V2": "final volume"},
       "M", "Ostwald (1893)", 7, ("mul", "div")),
    _L("compound_interest", "engineering",
       (1 + sp.Symbol("r")) ** sp.Symbol("t"),
       "A = P·(1+r)^t", {"r": "interest rate", "t": "time periods"},
       "1", "Bernoulli (1683)", 5, ("add", "pow"),
       "exponent is a variable — needs pow with variable exponent"),
]


def by_domain(domain: str) -> list[Law]:
    return [l for l in REGISTRY if l.domain == domain]


def by_difficulty(max_d: int) -> list[Law]:
    return sorted([l for l in REGISTRY if l.difficulty <= max_d],
                  key=lambda l: l.difficulty)


def get(name: str) -> Law:
    for l in REGISTRY:
        if l.name == name:
            return l
    raise KeyError(name)
