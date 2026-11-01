"""Expanded quantum observable candidates for 2-qubit systems.

Includes single-qubit Paulis, 2-qubit correlators, parity, and Bell
projectors. The pipeline's observable_selection ranks these by Jacobian
condition w.r.t. parameters.

Candidates are human-supplied. This is a wider candidate frame, not
autonomous observable invention.
"""
from __future__ import annotations
import numpy as np

I2 = np.eye(2, dtype=complex)
X2 = np.array([[0, 1], [1, 0]], dtype=complex)
Y2 = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z2 = np.array([[1, 0], [0, -1]], dtype=complex)


def _k(*ops):
    out = ops[0]
    for op in ops[1:]:
        out = np.kron(out, op)
    return out


SINGLE = {
    "X1": _k(X2, I2), "Y1": _k(Y2, I2), "Z1": _k(Z2, I2),
    "X2": _k(I2, X2), "Y2": _k(I2, Y2), "Z2": _k(I2, Z2),
}

CORRELATORS_2Q = {
    "XX": _k(X2, X2), "XY": _k(X2, Y2), "XZ": _k(X2, Z2),
    "YX": _k(Y2, X2), "YY": _k(Y2, Y2), "YZ": _k(Y2, Z2),
    "ZX": _k(Z2, X2), "ZY": _k(Z2, Y2), "ZZ": _k(Z2, Z2),
}

# Bell state projectors
def _bell_projector(ket):
    return np.outer(ket, ket.conj())

_PHI_PLUS  = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
_PHI_MINUS = np.array([1, 0, 0, -1], dtype=complex) / np.sqrt(2)
_PSI_PLUS  = np.array([0, 1, 1, 0], dtype=complex) / np.sqrt(2)
_PSI_MINUS = np.array([0, 1, -1, 0], dtype=complex) / np.sqrt(2)

BELL = {
    "P_Phi+": _bell_projector(_PHI_PLUS),
    "P_Phi-": _bell_projector(_PHI_MINUS),
    "P_Psi+": _bell_projector(_PSI_PLUS),
    "P_Psi-": _bell_projector(_PSI_MINUS),
}

PARITY = {
    "parity_XX": _k(X2, X2),
    "parity_ZZ": _k(Z2, Z2),
}

ALL = {**SINGLE, **CORRELATORS_2Q, **BELL, **PARITY}

INITIAL_STATE = np.array([1, 0, 1, 0], dtype=complex) / np.sqrt(2)  # |+0>


def evolve_expectation(J, h, t, obs_name, obs_dict=None):
    obs_dict = obs_dict or ALL
    def _H(J_, h_):
        return J_ * (_k(X2, X2) + _k(Y2, Y2) + _k(Z2, Z2)) + h_ * (_k(Z2, I2) + _k(I2, Z2))
    H = _H(J, h)
    evals, evecs = np.linalg.eigh(H)
    O = obs_dict[obs_name]
    v = evecs.conj().T @ INITIAL_STATE
    out = np.empty(len(t))
    for i, ti in enumerate(t):
        psi = evecs @ (np.exp(-1j * evals * ti) * v)
        out[i] = float((psi.conj() @ O @ psi).real)
    return out


def jacobian(obs_names, J, h, t, eps=1e-5, obs_dict=None):
    def stacked(J_, h_):
        return np.concatenate([evolve_expectation(J_, h_, t, n, obs_dict) for n in obs_names])
    base = stacked(J, h)
    jj = (stacked(J + eps, h) - base) / eps
    jh = (stacked(J, h + eps) - base) / eps
    return np.stack([jj, jh], axis=0)


COND_SUFFICIENT = 1e6


def rank_observables(J=0.7, h=0.4, t=None):
    t = t if t is not None else np.linspace(0, 10, 100)
    scores = []
    for name in ALL:
        Jac = jacobian([name], J, h, t)
        sv = np.linalg.svd(Jac, compute_uv=False)
        if len(sv) < 2 or sv[-1] < 1e-30:
            scores.append((name, 1, float("inf")))
        else:
            cond = float(sv[0] / sv[-1])
            rank = 1 if cond > COND_SUFFICIENT else 2
            scores.append((name, rank, cond))
    scores.sort(key=lambda s: (s[1] < 2, s[2]))
    return scores
