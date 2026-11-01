"""Observable selection by information content (restricted closing).

Ranks observable sets by Jacobian conditioning. Condition number is the
direct criterion: cond > 1e6 -> effective rank 1 (insufficient), else rank 2.
Numerical noise from finite-difference differentiation has a floor of
~eps * magnitude / step; cond thresholds are chosen to sit well above that
floor.

Restricted scope:
- Candidates are human-supplied.
- Ranking is by Jacobian condition, not full D-optimal design.
"""
from __future__ import annotations
import numpy as np

I2 = np.eye(2, dtype=complex)
X2 = np.array([[0, 1], [1, 0]], dtype=complex)
Y2 = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z2 = np.array([[1, 0], [0, -1]], dtype=complex)


def _H_zz(J, h):
    k = np.kron
    return J * (k(X2, X2) + k(Y2, Y2) + k(Z2, Z2)) + h * (k(Z2, I2) + k(I2, Z2))


PAULIS = {
    "X1": np.kron(X2, I2), "Y1": np.kron(Y2, I2), "Z1": np.kron(Z2, I2),
    "X2": np.kron(I2, X2), "Y2": np.kron(I2, Y2), "Z2": np.kron(I2, Z2),
}

INITIAL_STATE = np.array([1, 0, 1, 0], dtype=complex) / np.sqrt(2)


def evolve_expectation(J, h, t, obs_name):
    H = _H_zz(J, h)
    evals, evecs = np.linalg.eigh(H)
    O = PAULIS[obs_name]
    v = evecs.conj().T @ INITIAL_STATE
    out = np.empty(len(t))
    for i, ti in enumerate(t):
        psi = evecs @ (np.exp(-1j * evals * ti) * v)
        out[i] = float((psi.conj() @ O @ psi).real)
    return out


def jacobian(obs_names, J, h, t, eps=1e-5):
    def stacked(J_, h_):
        return np.concatenate([evolve_expectation(J_, h_, t, n) for n in obs_names])
    base = stacked(J, h)
    jj = (stacked(J + eps, h) - base) / eps
    jh = (stacked(J, h + eps) - base) / eps
    return np.stack([jj, jh], axis=0)


COND_SUFFICIENT_THRESHOLD = 1e6


def score_observables(candidate_sets, J, h, t):
    scores = []
    for names in candidate_sets:
        Jac = jacobian(list(names), J, h, t)
        sv = np.linalg.svd(Jac, compute_uv=False)
        if len(sv) < 2 or sv[-1] < 1e-30:
            rank, cond = 1, float("inf")
        else:
            cond = float(sv[0] / sv[-1])
            rank = 1 if cond > COND_SUFFICIENT_THRESHOLD else 2
        scores.append({"obs": tuple(names), "rank": rank, "cond": cond})
    scores.sort(key=lambda s: (s["rank"] < 2, s["cond"]))
    return scores
