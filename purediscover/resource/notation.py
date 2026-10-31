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
