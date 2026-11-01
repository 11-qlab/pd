"""Exact quantum-mechanical oracle with optional measurement noise."""
from __future__ import annotations

import csv
import itertools
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.linalg import expm

_PAULI = {
    "I": np.array([[1, 0], [0, 1]], dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


def _kron_n(ops):
    out = ops[0]
    for o in ops[1:]:
        out = np.kron(out, o)
    return out


def _pauli_string_op(pauli_str):
    return _kron_n([_PAULI[c] for c in pauli_str])


def _basis_state(bit_string):
    n = len(bit_string)
    idx = int(bit_string, 2)
    psi = np.zeros(2 ** n, dtype=complex)
    psi[idx] = 1.0
    return psi


@dataclass
class QuantumOracle:
    hamiltonian_spec: list
    n_qubits: int
    initial_state: str = "0"
    observable: str = "Z"
    time_param: str = "t"
    # --- measurement noise ----------------------------------------------
    noise_frac: float = 0.0        # multiplicative Gaussian noise sigma
    noise_seed: int | None = None  # reproducible noise stream

    def __post_init__(self):
        if not (1 <= self.n_qubits <= 6):
            raise ValueError(f"n_qubits must be in [1,6], got {self.n_qubits}")
        for name, pstr in self.hamiltonian_spec:
            if len(pstr) != self.n_qubits:
                raise ValueError(
                    f"Pauli string {pstr!r} does not match n_qubits={self.n_qubits}"
                )
        self._h_terms = [
            (name, _pauli_string_op(pstr))
            for name, pstr in self.hamiltonian_spec
        ]
        self._obs_op = _pauli_string_op(self.observable)
        if all(c in "01" for c in self.initial_state):
            self._psi0 = _basis_state(self.initial_state)
        elif self.initial_state == "+":
            p = np.array([1.0, 1.0], dtype=complex) / math.sqrt(2)
            self._psi0 = _kron_n([p] * self.n_qubits)
        else:
            raise ValueError(f"unsupported initial_state: {self.initial_state!r}")

        if not (0.0 <= self.noise_frac < 1.0):
            raise ValueError(f"noise_frac must be in [0, 1), got {self.noise_frac}")
        self._rng = np.random.default_rng(self.noise_seed)

    def _build_H(self, params):
        H = np.zeros((2 ** self.n_qubits, 2 ** self.n_qubits), dtype=complex)
        for name, op in self._h_terms:
            if name not in params:
                raise KeyError(f"missing Hamiltonian parameter {name!r}")
            H += params[name] * op
        return H

    def measure(self, params):
        """Exact <observable> after evolution, optionally with multiplicative noise.

        The noise models multiplicative measurement error (detector gain drift,
        calibration uncertainty) which is the dominant systematic in real
        qubit readout. Additive noise is available via noise_frac applied to
        the value directly before return, if you want that variant too.
        """
        if self.time_param not in params:
            raise KeyError(f"missing time parameter {self.time_param!r}")
        t = float(params[self.time_param])
        H = self._build_H(params)
        U = expm(-1j * H * t)
        psi = U @ self._psi0
        val = float(np.real(psi.conj() @ self._obs_op @ psi))
        if self.noise_frac > 0.0:
            # Multiplicative for nonzero values, additive for near-zero
            # (otherwise pure multiplicative can pin y=0 exactly, which is
            # unphysical — real detectors have additive shot noise floor).
            sigma = self.noise_frac
            if abs(val) > sigma * 5:
                val *= 1.0 + self._rng.normal(0.0, sigma)
            else:
                val += self._rng.normal(0.0, sigma)
        return val

    def to_csv(self, path, param_grid, *, target_name="y",
               noise_frac=None, seed=None):
        """Sweep the grid; if noise_frac is given it overrides the instance value."""
        old_nf = self.noise_frac
        old_rng = self._rng
        if noise_frac is not None:
            self.noise_frac = noise_frac
            self._rng = np.random.default_rng(seed if seed is not None
                                              else self.noise_seed)
        try:
            keys = list(param_grid.keys())
            rows = []
            for combo in itertools.product(*(param_grid[k] for k in keys)):
                p = dict(zip(keys, combo))
                y = self.measure(p)
                rows.append({**p, target_name: y})

            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=keys + [target_name])
                w.writeheader()
                for r in rows:
                    w.writerow({k: (f"{v:.10e}" if isinstance(v, float) else v)
                                for k, v in r.items()})
            return path
        finally:
            self.noise_frac = old_nf
            self._rng = old_rng
