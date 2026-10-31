"""Canonical literature queries.

Each value is either a string (single canonical phrase) or a list of
strings tried in order. Multi-query is needed when a single phrase
retrieves poor papers at a given source — try the second phrase only
if the first does not yield a title-phrase match.

Empirically tuned against litcheck (2026-10-31). Anything > 4 tokens
retrieves sparsely; use short phrases.
"""
from __future__ import annotations

LIT_QUERIES: dict[str, str | list[str]] = {
    # mechanics
    "newton_2":            "Newton's second law of motion",
    "kinetic_energy":      ["kinetic energy", "kinetic energy formula"],
    "gravitational_force": ["Newton's law of universal gravitation",
                            "universal gravitation"],
    "momentum":            "conservation of linear momentum",
    "angular_momentum":    "conservation of angular momentum",
    "centripetal_accel":   "centripetal acceleration",
    "hooke_law":           "Hooke's law",
    "pendulum_period":     "simple pendulum",
    "torque":              "torque",
    "mech_power":          "mechanical power",
    "work":                "mechanical work",
    "distance_kinematic":  "uniform motion",

    # electromagnetism
    "coulomb":             ["Coulomb's law", "electric charge Coulomb"],
    "ohm":                 "Ohm's law",
    "electric_power":      "electrical power",
    "lorentz_1d":          "Lorentz force",
    "lorentz_2d":          "Lorentz force",
    "biot_savart":         "Biot-Savart law",
    "capacitor_energy":    "capacitor energy",
    "inductor_energy":     "inductor energy",

    # thermodynamics
    "ideal_gas":           ["ideal gas law", "ideal gas equation"],
    "heat_capacity":       "specific heat capacity",
    "carnot_eff":          "Carnot efficiency",
    "stefan_boltzmann":    ["Stefan-Boltzmann law",
                            "Stefan-Boltzmann"],
    "boltzmann_entropy":   "Boltzmann entropy",

    # quantum — photon_energy retrieval is the known hard case
    "photon_energy":       ["photon energy",
                            "Planck relation",
                            "Planck-Einstein relation",
                            "photoelectric effect"],
    "de_broglie":          "de Broglie wavelength",
    "hydrogen_levels":     "Bohr model",
    "bohr_radius":         "Bohr radius",

    # relativity
    "emc2":                ["mass-energy equivalence",
                            "E=mc^2",
                            "Einstein mass-energy"],
    "energy_momentum":     "relativistic energy-momentum",

    # optics
    "magnification":       "lens magnification",
    "inverse_square":      "inverse square law",

    # waves
    "wave_speed":          "wave equation",
    "period_freq":         "period frequency",

    # fluids
    "flow_rate":           "volumetric flow rate",
    "pressure":            "pressure",
    "hydrostatic":         "hydrostatic pressure",
    "reynolds":            "Reynolds number",

    # chemistry
    "rate_law_bimol":      "bimolecular reaction rate",
    "ph":                  "pH definition",

    # biology
    "michaelis_menten":    ["Michaelis-Menten kinetics",
                            "Michaelis constant"],
    "logistic_growth":     "logistic population growth",

    # mathematics
    "area_rect":           "area of a rectangle",
    "area_circle":         "area of a circle",
    "volume_sphere":       "volume of a sphere",
    "pythagoras_sq":       "Pythagorean theorem",
    "hypotenuse":          "Pythagorean theorem",
    "triangle_area":       "area of a triangle",

    # probability
    "bayes":               ["Bayes theorem", "Bayes' theorem",
                            "Bayes formula"],
    "variance":            "variance",
    "covariance":          "covariance",

    # engineering
    "rect_perimeter":      "perimeter of a rectangle",
    "cylinder_volume":     "volume of a cylinder",
    "dilution":            "dilution equation",
    "compound_interest":   "compound interest",
}


def get_queries(law_name: str) -> list[str]:
    """Return list of canonical phrases to try, in order."""
    v = LIT_QUERIES.get(law_name, [])
    if isinstance(v, str):
        return [v] if v else []
    return list(v)


def get_query(law_name: str, fallback: str = "") -> str:
    """Back-compat: return first canonical phrase."""
    qs = get_queries(law_name)
    return qs[0] if qs else fallback


__all__ = ["LIT_QUERIES", "get_query", "get_queries"]
