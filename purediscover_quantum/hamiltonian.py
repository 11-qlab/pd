"""2-qubit Heisenberg Hamiltonian and observable traces.

H = J*(X⊗X + Y⊗Y + Z⊗Z) + h*(Z⊗I + I⊗Z)

Key result: starting in |01>, the observable <Z_1>(t) = cos(4*J*t),
independent of h. This is the cleanest Hamiltonian-learning target:
one frequency, one coupling, exact elementary law.

Starting in |00>: <Z_1>(t) = 1 (eigenstate, no evolution). Not useful.

Starting in |+0>: <Z_1>(t) depends on both J and h; use expm for exact.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.linalg import expm

_PAULI = {
    "I": np.array([[1, 0], [0, 1]], dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


def _kron2(a, b):
    return np.kron(_PAULI[a], _PAULI[b])


def heisenberg_2q(J: float, h: float = 0.0) -> np.ndarray:
    return (J * (_kron2("X", "X") + _kron2("Y", "Y") + _kron2("Z", "Z"))
            + h * (_kron2("Z", "I") + _kron2("I", "Z")))


def _state(bit_string: str) -> np.ndarray:
    if len(bit_string) != 2:
        raise ValueError(f"need 2-bit state, got {bit_string!r}")
    if all(c in "01" for c in bit_string):
        idx = int(bit_string, 2)
        psi = np.zeros(4, dtype=complex)
        psi[idx] = 1.0
        return psi
    if bit_string == "+0":
        p = np.array([1.0, 1.0], dtype=complex) / math.sqrt(2)
        z = np.array([1.0, 0.0], dtype=complex)
        return np.kron(p, z)
    raise ValueError(f"unsupported state {bit_string!r}")


def trace_Z1(t: float, J: float, h: float = 0.0,
             initial: str = "01") -> float:
    """<Z_1>(t) for the 2-qubit Heisenberg Hamiltonian."""
    if initial == "01":
        return math.cos(4.0 * J * t)
    if initial == "00":
        return 1.0
    H = heisenberg_2q(J, h)
    psi0 = _state(initial)
    psi_t = expm(-1j * H * t) @ psi0
    Z1 = _kron2("Z", "I")
    return float(np.real(psi_t.conj() @ Z1 @ psi_t))


def make_trace_data(J: float, h: float = 0.0,
                    initial: str = "01",
                    n: int = 50,
                    t_periods: float = 2.5,
                    noise_frac: float = 0.02,
                    seed: int = 0):
    """Return data as [({'t': t}, y_noisy)], where y = <Z_1>(t) + noise.

    t_periods: how many oscillation periods to cover (period = pi/(2*J)).
    """
    rng = np.random.default_rng(seed)
    period = math.pi / (2.0 * abs(J)) if abs(J) > 1e-9 else math.pi
    t_max = t_periods * period
    ts = np.linspace(0.02, t_max, n)
    data = []
    for t in ts:
        y = trace_Z1(float(t), J, h, initial)
        if noise_frac > 0:
            y = y * (1.0 + rng.normal(0.0, noise_frac))
        data.append(({"t": float(t)}, float(y)))
    return data
