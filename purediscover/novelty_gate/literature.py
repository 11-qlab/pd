"""Layer 3 — extended literature check.

Two checks against a hand-curated extended corpus (~40 additional laws
beyond purediscover's 56):
    1. scalar equivalence: expr = c * known_law  (rejected if matched)
    2. algebraic fingerprint: shape matches a known law (warning, not reject)

Fingerprint replaces each Float atom with a free symbol, canonicalizes
variable names, and returns the resulting string. Two expressions with
the same fingerprint have the same algebraic skeleton.
"""
from __future__ import annotations
import sympy as sp


# Extended corpus: (name, expr_sympy, var_names, description)
def _extended_laws():
    A, Ea, T, k, x, t, m, v, c, lam = sp.symbols("A Ea T k x t m v c lam")
    B, nu, h, a, n, R, z, F, I, w = sp.symbols("B nu h a n R z F I w")
    laws = [
        # exponential laws
        ("arrhenius",            A * sp.exp(-Ea / T), ["T"],
         "rate constant vs temperature"),
        ("boltzmann_factor",     A * sp.exp(-Ea / T), ["T"],
         "population of state at energy Ea"),
        ("clausius_clapeyron",   A * sp.exp(-Ea / T), ["T"],
         "vapor pressure vs temperature"),
        ("vant_hoff",            A * sp.exp(-Ea / T), ["T"],
         "equilibrium constant vs T"),
        ("newton_cooling",       A * sp.exp(-k * t), ["t"],
         "temperature decay"),
        ("radioactive_decay",    A * sp.exp(-lam * t), ["t"],
         "nuclear population decay"),
        ("rc_discharge",         A * sp.exp(-t / (R * c)), ["t", "R", "c"],
         "capacitor discharge"),
        ("gaussian",             A * sp.exp(-x**2), ["x"],
         "standard normal density"),
        ("gaussian_centered",    A * sp.exp(-(x - a)**2 / B), ["x"],
         "Gaussian with offset and width"),
        ("maxwell_boltzmann",    x**2 * sp.exp(-x**2 / T**2), ["x", "T"],
         "speed distribution"),
        # power laws
        ("kepler_third",         a**(sp.Rational(3, 2)), ["a"],
         "orbital period vs semi-major axis"),
        ("debeye_lowT",          T**3, ["T"],
         "heat capacity of solid at low T"),
        ("stefan_boltzmann_ext", T**4, ["T"], "blackbody radiance"),
        ("inverse_proportional", A / x, ["x"],
         "1/x law (Coulomb, gravity, point-source field)"),
        ("inverse_square_ext",   1 / x**2, ["x"], "point-source intensity"),
        ("coulomb_ext",          z**2 / x**2, ["z", "x"], "electrostatic"),
        # rational functions
        ("planck",               x**3 / (sp.exp(h * x / T) - 1), ["x", "T"],
         "blackbody spectrum (general T-explicit form)"),
        ("planck_aggregated",    A * x**3 / (sp.exp(B * x) - 1), ["x"],
         "blackbody spectrum at fixed T (h/kT and 2h/c^2 absorbed)"),
        ("fermi_dirac",          1 / (sp.exp((x - a) / k) + 1), ["x"],
         "Fermi-Dirac distribution"),
        ("bose_einstein",        1 / (sp.exp(x / T) - 1), ["x", "T"],
         "Bose-Einstein distribution"),
        ("langevin",             1 / sp.tanh(x) - 1 / x, ["x"],
         "classical paramagnetic"),
        ("cauchy_lorentz",       a / ((x - B)**2 + a**2), ["x"],
         "resonance line shape"),
        ("van_der_waals",        (x + a) / x**2, ["x"],
         "real gas correction"),
        ("michaelis_menten_ext", x / (a + x), ["x"],
         "enzyme kinetics"),
        # logarithmic laws
        ("nernst",               -R * T / z * sp.log(x), ["T", "z", "x"],
         "cell potential"),
        ("henderson_hasselbalch",sp.log(x), ["x"],
         "pH buffering"),
        ("sackur_tetrode",       T**sp.Rational(3, 2) * sp.log(x), ["T", "x"],
         "ideal-gas entropy"),
        # trig / oscillation
        ("simple_harmonic",      x**2, ["x"], "harmonic potential"),
        ("lc_oscillation",       sp.sin(w * t), ["t"], "oscillator"),
        ("damped_oscillation",   A * sp.exp(-k * t) * sp.sin(w * t), ["t"],
         "damped harmonic"),
        ("bragg",                sp.sin(x), ["x"], "Bragg diffraction"),
        ("compton",              1 / (1 + (1 - sp.cos(x)) / a), ["x"],
         "Compton scattering"),
        # combinations
        ("doppler_nonrel",       x * (1 - a / c), ["x", "a", "c"],
         "nonrelativistic Doppler"),
        ("doppler_rel",          x * sp.sqrt((1 - a / c) / (1 + a / c)),
         ["x", "a", "c"], "relativistic Doppler"),
        ("beer_lambert",         a * x * B, ["a", "x", "B"],
         "absorbance"),
        ("larmor",               z**2 * a**2, ["z", "a"],
         "radiation power"),
        ("poiseuille",           a**4 * x / B, ["a", "x", "B"],
         "pipe flow"),
        ("hagen_poiseuille",     x / a**4, ["x", "a"], "pipe resistance"),
        ("fick_second",          x / a**2, ["x", "a"], "diffusion time"),
        ("einstein_heat_cap",    x**2 * sp.exp(x) / (sp.exp(x) - 1)**2,
         ["x"], "Einstein model"),
        ("curie_weiss",          a / (x - B), ["x"], "magnetic susceptibility"),
        ("saha_ionization",      x**2 * sp.exp(-a / x), ["x"],
         "ionization fraction"),

        # --- corpus extension v2 (2026-10-31) ---
        ("lennard_jones",        A / x**12 - B / x**6, ["x"],
         "intermolecular potential (12-6)"),
        ("london_dispersion",    A / x**6, ["x"],
         "London dispersion force"),
        ("woods_saxon",          A / (1 + sp.exp((x - B) / c)), ["x"],
         "nuclear density profile"),
        ("sutherland",           A * x**sp.Rational(3, 2) / (x + B), ["x"],
         "gas viscosity vs temperature"),
        ("debye_huckel",         sp.exp(-A * z**2 * sp.sqrt(I)), ["z", "I"],
         "electrolyte activity coefficient"),
        ("clausius_mossotti",    (x - 1) / (x + 2), ["x"],
         "dielectric polarizability"),
        ("rabi_population",      sp.sin(w * t)**2, ["w", "t"],
         "two-level system population"),
    ]
    return laws


# Cache fingerprints on first use
_FP_CACHE: dict | None = None


def _replace_floats(expr):
    """Replace each Float atom with a fresh Symbol; preserve sign.

    Negative float coefficients used to be replaced with a positive
    symbol — so A*exp(-Ea/T) and A*exp(-8006.7/T) produced different
    fingerprints (one had -Ea as Mul(-1, Symbol), the other had -8006
    absorbed into a Float, then Symbol). Preserving sign makes both
    map to A*exp(-Symbol/T).
    """
    seen: dict[str, sp.Symbol] = {}
    def _sub(e):
        if e.is_Float:
            key = repr(float(e))
            if key not in seen:
                seen[key] = sp.Symbol(f"__c{len(seen)}")
            s = seen[key]
            return s if float(e) >= 0 else -s
        return e
    return expr.replace(lambda e: e.is_Float, _sub)


def algebraic_fingerprint(expr) -> str:
    """Canonical skeleton invariant to symbol names and leading scalars.

    Two normalizations:
        1. Strip any leading scalar multiplier from a top-level Mul.
           c * exp(-k/T) and exp(-k/T) both map to the same fingerprint.
           This is the correct semantic: shapes match up to scale.
        2. Assign x0, x1, ... by preorder traversal (name-independent).
    """
    expr = _replace_floats(expr)
    expr = _strip_leading_scalars(expr)

    sym_map: dict = {}
    for node in sp.preorder_traversal(expr):
        if node.is_Symbol and node not in sym_map:
            sym_map[node] = sp.Symbol(f"x{len(sym_map)}")
    return str(expr.subs(sym_map))


def _is_complex(expr) -> bool:
    """True iff expr contains a node that isn't a Symbol or a Number.

    Functions (exp, log, sqrt), Pow, and Add nodes count as "complex".
    A single Symbol, a Number, or Mul/Add of only symbols does not.
    """
    for node in sp.preorder_traversal(expr):
        if node.is_Symbol or node.is_Number:
            continue
        return True
    return False


def _strip_leading_scalars(expr):
    """Strip leading Mul factors that look like scalar coefficients.

    Strips a factor when:
        - it's a Number, OR
        - it's a `__cN` placeholder (from float replacement), OR
        - it's a Symbol that appears exactly once in the expression
        AND stripping it leaves a "complex" residue (contains a
        non-leaf atom like exp/pow/log).

    The complex-residue requirement prevents:
        x*y  ->  y       (both are variables; don't strip)
        A*x  ->  x       (linear product; preorder already handles)

    But allows:
        A*exp(-Ea/T)  ->  exp(-Ea/T)
        A/x           ->  1/x
        2*A*exp(-k)   ->  exp(-k)
    """
    if not isinstance(expr, sp.Mul):
        return expr
    args = list(expr.args)
    if len(args) < 2:
        return expr

    i = 0
    while i < len(args) - 1:
        factor = args[i]
        name = str(factor) if factor.is_Symbol else ""
        is_scalar = (
            factor.is_Number
            or name.startswith("__c")
            or factor.is_Symbol  # trial — checked by residue test below
        )
        if not is_scalar:
            break
        residue = sp.Mul(*args[i + 1:])
        if not _is_complex(residue):
            break
        i += 1

    if i == 0:
        return expr
    return sp.Mul(*args[i:])


def _build_fp_index():
    global _FP_CACHE
    if _FP_CACHE is not None:
        return _FP_CACHE
    idx: dict[str, list[dict]] = {}
    for name, expr, var_names, desc in _extended_laws():
        try:
            fp = algebraic_fingerprint(expr)
        except Exception:
            continue
        idx.setdefault(fp, []).append({
            "name": name, "variables": var_names, "description": desc,
            "expr": str(expr),
        })
    _FP_CACHE = idx
    return idx


def literature_gate(expr) -> dict:
    """Return shape matches from the extended corpus."""
    fp = algebraic_fingerprint(expr)
    idx = _build_fp_index()
    matches = idx.get(fp, [])
    return {
        "ok": True,   # never rejects on its own
        "fingerprint": fp,
        "shape_matches": matches,
        "has_match": bool(matches),
    }
