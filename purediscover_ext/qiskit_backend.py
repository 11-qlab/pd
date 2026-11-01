from __future__ import annotations
import numpy as np
from scipy.linalg import expm

I2 = np.eye(2, dtype=complex)
X2 = np.array([[0, 1], [1, 0]], dtype=complex)
Y2 = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z2 = np.array([[1, 0], [0, -1]], dtype=complex)


def _k(*ops):
    out = ops[0]
    for op in ops[1:]:
        out = np.kron(out, op)
    return out


def _H_zz_matrix(J, h):
    return J * (_k(X2, X2) + _k(Y2, Y2) + _k(Z2, Z2)) + h * (_k(Z2, I2) + _k(I2, Z2))


def _obs_matrix(name):
    table = {
        "X1": _k(X2, I2), "Y1": _k(Y2, I2), "Z1": _k(Z2, I2),
        "X2": _k(I2, X2), "Y2": _k(I2, Y2), "Z2": _k(I2, Z2),
        "XX": _k(X2, X2), "YY": _k(Y2, Y2), "ZZ": _k(Z2, Z2),
    }
    return table[name]


PLUS0 = np.array([1, 0, 1, 0], dtype=complex) / np.sqrt(2)


def exact_expectation(J, h, t_array, obs_name, initial=PLUS0):
    H = _H_zz_matrix(J, h)
    O = _obs_matrix(obs_name)
    evals, evecs = np.linalg.eigh(H)
    v = evecs.conj().T @ initial
    out = np.empty(len(t_array))
    for i, ti in enumerate(t_array):
        psi = evecs @ (np.exp(-1j * evals * ti) * v)
        out[i] = float((psi.conj() @ O @ psi).real)
    return out


def shots_expectation(J, h, t_array, obs_name, shots=1024, seed=None):
    from qiskit import QuantumCircuit, transpile
    from qiskit_aer import AerSimulator
    from qiskit.circuit.library import UnitaryGate

    H = _H_zz_matrix(J, h)
    sim = AerSimulator(seed_simulator=seed)
    out = np.empty(len(t_array))

    for i, ti in enumerate(t_array):
        U = expm(-1j * H * ti)
        qc = QuantumCircuit(2, 2)
        qc.initialize(PLUS0, [0, 1])
        qc.append(UnitaryGate(U), [0, 1])
        # Convention: numpy "Z1" = Z on first kron factor = qiskit qubit 1
        # (state vector index MSB). numpy "Z2" = second factor = qubit 0 (LSB).
        if obs_name == "Z1":
            qc.measure([1], [0])
        elif obs_name == "X1":
            qc.h(1)
            qc.measure([1], [0])
        elif obs_name == "Z2":
            qc.measure([0], [0])
        elif obs_name in ("ZZ", "XX"):
            if obs_name == "XX":
                qc.h(0)
                qc.h(1)
            qc.measure([0, 1], [0, 1])
        else:
            raise ValueError("shots mode not implemented for " + obs_name)

        compiled = transpile(qc, sim)
        counts = sim.run(compiled, shots=shots).result().get_counts()

        exp_val = 0.0
        for bits, c in counts.items():
            if obs_name in ("Z1", "X1"):
                b0 = int(bits[-1])
                exp_val += (1 if b0 == 0 else -1) * c
            elif obs_name == "Z2":
                b1 = int(bits[-2])
                exp_val += (1 if b1 == 0 else -1) * c
            elif obs_name in ("ZZ", "XX"):
                b0 = int(bits[-1])
                b1 = int(bits[-2])
                s0 = 1 if b0 == 0 else -1
                s1 = 1 if b1 == 0 else -1
                exp_val += (s0 * s1) * c
        out[i] = exp_val / shots

    return out


def fit_J_h_from_traces(J_init, h_init, t_z1, z1_obs, t_x1, x1_obs):
    from scipy.optimize import minimize

    def loss(p):
        J_, h_ = p
        if J_ <= 0 or h_ <= 0:
            return 1e10
        zp = exact_expectation(J_, h_, t_z1, "Z1")
        xp = exact_expectation(J_, h_, t_x1, "X1")
        return float(np.mean((zp - z1_obs) ** 2) + np.mean((xp - x1_obs) ** 2))

    best = (np.inf, J_init, h_init)
    for J0 in np.linspace(0.05, 2.0, 15):
        for h0 in np.linspace(0.05, 2.0, 15):
            l = loss((J0, h0))
            if l < best[0]:
                best = (l, J0, h0)
    res = minimize(loss, [best[1], best[2]], method="Nelder-Mead",
                   options={"xatol": 1e-7, "fatol": 1e-10, "maxiter": 2000})
    return float(res.x[0]), float(res.x[1])
