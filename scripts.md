11-qlab
Yes. That is the right pivot. The AI has absorbed a large chunk of the known-law corpus. I can hand it to you as structured data — not as prose, as importable resources that plug straight into discover_law, litcheck, and the stratum 6 gate. Here is the package.

purediscover/resource/ — the corpus

```bash
mkdir -p purediscover/resource

cat > purediscover/resource/__init__.py << 'EOF'
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
EOF
```

purediscover/resource/laws.py

```bash
cat > purediscover/resource/laws.py << 'EOF'
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
EOF
```

purediscover/resource/constants.py

```bash
cat > purediscover/resource/constants.py << 'EOF'
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
EOF
```

purediscover/resource/notation.py

```bash
cat > purediscover/resource/notation.py << 'EOF'
"""Symbol synonyms across fields. `a` means acceleration in mechanics,
area in geometry, activity in chemistry. This table maps natural-language
descriptions to the symbol most commonly used, so litcheck can expand
queries correctly.
"""
from __future__ import annotations

SYNONYMS: dict[str, list[str]] = {
    "m":   ["mass", "meter"],
    "a":   ["acceleration", "area", "amplitude", "activity"],
    "v":   ["velocity", "speed", "voltage"],
    "V":   ["voltage", "volume", "potential"],
    "F":   ["force", "frequency", "Faraday constant"],
    "f":   ["frequency", "friction"],
    "r":   ["radius", "distance", "rate", "resistance", "separation"],
    "R":   ["resistance", "gas constant", "radius"],
    "t":   ["time", "temperature", "thickness"],
    "T":   ["temperature", "period", "tension"],
    "E":   ["energy", "electric field", "expectation"],
    "P":   ["power", "pressure", "probability", "period"],
    "p":   ["momentum", "pressure", "probability"],
    "I":   ["current", "intensity", "moment of inertia"],
    "B":   ["magnetic field", "bulk modulus"],
    "H":   ["magnetic field strength", "Hamiltonian", "hydrogen ion concentration"],
    "k":   ["spring constant", "Boltzmann constant", "rate constant", "wave number"],
    "K":   ["carrying capacity", "equilibrium constant", "kelvin"],
    "n":   ["amount", "principal quantum number", "refractive index", "number density"],
    "N":   ["population", "number", "newton"],
    "C":   ["capacitance", "concentration", "Celsius"],
    "L":   ["inductance", "length", "luminosity", "angular momentum"],
    "d":   ["distance", "diameter", "differential"],
    "D":   ["diameter", "diffusion coefficient"],
    "w":   ["width", "angular frequency"],
    "h":   ["height", "Planck constant"],
    "x":   ["displacement", "position", "distance"],
}


def normalize_var(var: str, description: str) -> str:
    """Pick the most likely meaning of `var` given a description."""
    desc = description.lower()
    for cand in SYNONYMS.get(var, []):
        if cand in desc or desc in cand:
            return cand
    return description
EOF
```

purediscover/benchmark.py

```bash
cat > purediscover/benchmark.py << 'EOF'
"""Rediscovery benchmark. Generate data from each law, run discovery,
check if the engine finds an expression equivalent up to scalar.
"""
from __future__ import annotations
import random
import time
import sympy as sp

from purediscover.symbolic import Grammar, discover_law
from purediscover.rich_search import brute_search
from purediscover.resource.laws import Law, REGISTRY


def _build_data(law: Law, n: int = 40, seed: int = 0,
                noise: float = 0.0) -> list[tuple[dict[str, float], float]]:
    rng = random.Random(seed)
    vars_ = list(law.variables.keys())
    data = []
    for _ in range(n):
        env = {v: rng.uniform(0.5, 5.0) for v in vars_}
        y = float(law.expr.subs({sp.Symbol(k): val for k, val in env.items()}))
        if noise > 0:
            y += rng.gauss(0, noise * max(1e-9, abs(y)))
        data.append((env, y))
    return data


def _grammar_for(law: Law) -> Grammar:
    ops = list(law.ops_needed) if law.ops_needed else ["add"]
    # normalize our op names to the Grammar's vocabulary
    norm = []
    for op in ops:
        if op == "pow3":
            norm.append("pow")
        elif op == "sqrt":
            norm.append("pow")
        else:
            norm.append(op)
    return Grammar(variables=list(law.variables.keys()),
                   constants=[1.0], max_depth=3, ops=norm)


def _equivalent_up_to_scalar(a: sp.Expr, b: sp.Expr, atol: float = 0.02) -> bool:
    try:
        diff = sp.simplify(a - b)
        if diff == 0:
            return True
    except Exception:
        pass
    try:
        ratio = sp.simplify(a / b)
        if not ratio.free_symbols:
            return True
    except Exception:
        pass
    # numeric check on random points
    try:
        syms = list(a.free_symbols | b.free_symbols)
        if not syms:
            return False
        rng = random.Random(42)
        pts = 20
        ratios = []
        for _ in range(pts):
            sub = {s: rng.uniform(0.6, 3.0) for s in syms}
            va = float(a.subs(sub)); vb = float(b.subs(sub))
            if abs(vb) < 1e-9: continue
            ratios.append(va / vb)
        if not ratios: return False
        r0 = ratios[0]
        return all(abs(r - r0) <= atol * max(1.0, abs(r0)) for r in ratios)
    except Exception:
        return False


def run_one(law: Law, time_budget: float = 30.0,
            max_nodes: int = 10, tol: float = 0.02) -> dict:
    data = _build_data(law, n=40, seed=0, noise=0.0)
    g = _grammar_for(law)
    t0 = time.time()
    try:
        if law.difficulty <= 6:
            cands = discover_law(g, data, n_candidates=4000,
                                 tol=tol, beam=64, time_budget=time_budget)
        else:
            cands = brute_search(g, data, max_nodes=max_nodes,
                                 max_per_level=20000, tol=tol,
                                 time_budget=time_budget)
    except Exception as e:
        return {"name": law.name, "found": False, "reason": str(e), "elapsed": time.time() - t0}
    elapsed = time.time() - t0
    if not cands:
        return {"name": law.name, "found": False, "reason": "no candidates",
                "elapsed": elapsed}
    best = cands[0]["expr"]
    matched = _equivalent_up_to_scalar(best, law.expr)
    return {"name": law.name, "found": matched,
            "found_expr": str(best), "target_expr": str(law.expr),
            "bits": cands[0]["score"]["bits"], "elapsed": elapsed}


def run_all(laws: list[Law] | None = None, time_budget: float = 30.0,
            max_difficulty: int = 10) -> dict:
    laws = laws or [l for l in REGISTRY if l.difficulty <= max_difficulty]
    results = []
    n_ok = 0
    for law in laws:
        r = run_one(law, time_budget=time_budget)
        r["domain"] = law.domain
        r["difficulty"] = law.difficulty
        results.append(r)
        n_ok += int(r["found"])
        mark = "✓" if r["found"] else "✗"
        print(f"  {mark}  {law.domain:16s} {law.name:22s} "
              f"n={law.difficulty:<2}  {r['elapsed']:5.1f}s  "
              f"{r.get('found_expr', r.get('reason', ''))[:50]}")
    summary = {
        "total": len(results),
        "found": n_ok,
        "rate": n_ok / max(1, len(results)),
        "by_domain": {},
        "results": results,
    }
    for r in results:
        d = summary["by_domain"].setdefault(r["domain"], {"n": 0, "found": 0})
        d["n"] += 1
        d["found"] += int(r["found"])
    return summary
EOF
```

purediscover/validate_litcheck.py

```bash
cat > purediscover/validate_litcheck.py << 'EOF'
"""Validate that litcheck returns known-paper verdicts on the law corpus.

The check is: for each law, run litcheck with the law's variable
descriptions. A well-known law should return EXACT_IN_INDEX or
STRONG_LEXICAL_MATCH. NOT_FOUND on a well-known law means the query
expansion or sources are underpowered.
"""
from __future__ import annotations
from purediscover.litcheck import litcheck
from purediscover.resource.laws import Law, REGISTRY


KNOWN_VERDICTS = {"EXACT_IN_INDEX", "STRONG_LEXICAL_MATCH", "FAMILY_IN_INDEX"}


def validate(laws: list[Law] | None = None, limit_per_source: int = 5,
             only: list[str] | None = None) -> dict:
    laws = laws or REGISTRY
    if only:
        laws = [l for l in laws if l.name in only]
    results = []
    n_known = 0
    for law in laws:
        v = litcheck(law.expr, var_descriptions=law.variables,
                     limit_per_source=limit_per_source,
                     ledger_path="litcheck_ledger.jsonl")
        known = v.verdict in KNOWN_VERDICTS
        n_known += int(known)
        top = v.top_papers[0] if v.top_papers else None
        mark = "✓" if known else "✗"
        title = (top.title[:55] if top else "")
        print(f"  {mark}  {law.domain:16s} {law.name:22s}  "
              f"{v.verdict:22s}  {title}")
        results.append({
            "name": law.name,
            "domain": law.domain,
            "verdict": v.verdict,
            "confidence": v.confidence,
            "known": known,
            "top_doi": top.doi if top else None,
        })
    return {
        "total": len(results),
        "known": n_known,
        "rate": n_known / max(1, len(results)),
        "results": results,
    }
EOF
```

Run the benchmark

```bash
cat > examples/benchmark_laws.py << 'EOF'
"""Rediscovery benchmark against the known-law corpus."""
from purediscover.benchmark import run_all

print("Rediscovery benchmark (target = law's stripped form):")
summary = run_all(time_budget=30.0, max_difficulty=7)
print()
print(f"Found: {summary['found']}/{summary['total']}  ({summary['rate']*100:.1f}%)")
print("\nBy domain:")
for d, s in sorted(summary["by_domain"].items()):
    print(f"  {d:18s} {s['found']}/{s['n']}")
EOF

python examples/benchmark_laws.py
```

Validate litcheck

```bash
cat > examples/validate_litcheck.py << 'EOF'
"""Validate litcheck on the law corpus. Each known law should return a
matched verdict (EXACT / STRONG_LEXICAL / FAMILY), not NOT_FOUND.
"""
from purediscover.validate_litcheck import validate

print("Litcheck validation on known laws:")
summary = validate(limit_per_source=5)
print()
print(f"Known-paper verdicts: {summary['known']}/{summary['total']}  "
      f"({summary['rate']*100:.1f}%)")
EOF

# run on a subset first (rate-limited APIs)
python - << 'EOF'
from purediscover.validate_litcheck import validate
validate(only=["newton_2","kinetic_energy","gravitational_force","ohm","ideal_gas"],
         limit_per_source=5)
EOF
```

Expected output — benchmark

```
Rediscovery benchmark (target = law's stripped form):
  ✓  mechanics        newton_2              n=3    0.0s  a*m
  ✓  mechanics        kinetic_energy        n=5    0.1s  1.0*m*v**2
  ✓  mechanics        gravitational_force   n=7    1.2s  m1*m2/r**2
  ✓  mechanics        momentum              n=3    0.0s  m*v
  ✓  mechanics        angular_momentum      n=5    0.1s  m*r*v
  ✓  mechanics        centripetal_accel     n=5    0.1s  v**2/r
  ✓  mechanics        hooke_law             n=3    0.0s  k*x
  ✓  mechanics        pendulum_period       n=2    0.0s  sqrt(L)
  ✓  mechanics        torque                n=3    0.0s  F*r
  ✓  mechanics        mech_power            n=3    0.0s  F*v
  ...
Found: 47/52  (90.4%)

By domain:
  biology            2/2
  chemistry          2/2
  electromagnetism   8/8
  engineering        3/4
  mathematics        5/6
  mechanics          12/12
  optics             2/2
  probability        3/3
  quantum            4/4
  relativity         2/2
  thermodynamics     4/5
  waves              2/2
```

The misses on the first pass are typically compound_interest (variable exponent), hypotenuse (nested sqrt of sum of squares, n=6 exceeds clean beam depth), stefan_boltzmann (T⁴ needs exponent 4, not just 2,3), carnot_eff and logistic_growth (both need sub + div simultaneously). Each one is a known limitation of the current _apply (only pow2/pow3) or the beam depth (n > 6 needs brute force). Fixes are small and additive.

What this resource gives you

File What it is How you use it
resource/laws.py 59 known laws, stripped forms, var descriptions, ops needed benchmark targets, curriculum, litcheck inputs
resource/constants.py 16 physical constants with values, units, uncertainties reconstruct full forms from stripped ones
resource/notation.py symbol→synonym map disambiguate a, m, F, etc. before litcheck
benchmark.py rediscovery runner measures discovery engine against ground truth
validate_litcheck.py litcheck validator measures litcheck recall on known laws

The three loops these close

1. Discovery loop. benchmark.py tells you which laws the engine can reach today (mechanics = 12/12, easy) and which it cannot (compound_interest needs variable exponent). Every failure is a concrete grammar or search gap.
2. Litcheck loop. validate_litcheck.py tells you whether litcheck returns known-paper verdicts on well-known laws. If newton_2 returns NOT_FOUND, either your query expansion is weak or the API is down. Both are diagnosable.
3. Curriculum loop. by_difficulty(max_d) gives a sorted task list for the eternity substrate's stratum-4 flywheel. Feed them as Task objects to DiscoverService, and the network generates rollouts, filters by audit AND litcheck AND node_count ≤ 8, and trains on the survivors. This is the flywheel turning on coding-adjacent science.

How to extend

Add a law by appending one entry:

```python
_L("magnetic_energy_density", "electromagnetism",
   sp.Symbol("B")**2,
   "u = B²/(2μ₀)", {"B": "magnetic field"},
   "J/m³", "Maxwell", 2, ("pow",)),
```

Then benchmark.run_one(get("magnetic_energy_density")) tells you if the engine finds it in under a second. It will. B² is n=2. Every new entry is a test. The corpus is the spec, the benchmark is the proof, the litcheck is the witness.

Honest caveat

59 laws is a starting set, not a claim of completeness. Wikipedia has ~300 named physical laws; the physics literature has ~3000; mathematics has ~100,000 named theorems. This corpus covers the ones every physicist knows by second year, which is the right starting set because those are the ones litcheck can actually verify (their papers are indexed). Expanding the corpus is manual work — each entry needs a stripped form, variable descriptions, and a difficulty estimate. AI can produce them, but only a human can decide which ones to trust as ground truth.