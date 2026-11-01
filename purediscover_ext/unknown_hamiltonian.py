"""Unknown-Hamiltonian discovery on the 2-qubit ZZ model.

H = J*(XX + YY + ZZ) + h*(ZI + IZ)

Note: contrary to a first guess, with initial state |+0>=(|0>+|1>)/sqrt(2)*|0>,
the observable <Z1>(t) = 1/2 - 1/2*cos(2Jt) does NOT depend on h. Reason:
|00> is an eigenstate with all +1 Z-projections, so its amplitude drops out
of <Z1>; the {|01>,|10>} block is h-free. The |01> initial state also only
reveals J.

What DOES depend on both:
    <X1>(t) = <X x I>(t) = 1/2 [cos(2h t) + cos(2(h-J) t)]
which is what we use, jointly with <Z1>(t) to break a J <-> 2h - J symmetry.
"""
from __future__ import annotations
import numpy as np

# ---------------------------------------------------------------- Paulis
I2 = np.eye(2, dtype=complex)
X2 = np.array([[0, 1], [1, 0]], dtype=complex)
Y2 = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z2 = np.array([[1, 0], [0, -1]], dtype=complex)


def _kron(*ops):
    out = ops[0]
    for op in ops[1:]:
        out = np.kron(out, op)
    return out


XX, YY, ZZ = _kron(X2, X2), _kron(Y2, Y2), _kron(Z2, Z2)
ZI, IZ = _kron(Z2, I2), _kron(I2, Z2)
X1 = ZI @ np.zeros_like(ZI) + _kron(X2, I2)  # X x I
Z1 = ZI
PLUS0 = np.array([1, 0, 1, 0], dtype=complex) / np.sqrt(2)


def _H(J, h):
    return J * (XX + YY + ZZ) + h * (ZI + IZ)


# ---------------------------------------------------------------- simulation
def trace(J, h, t_array, obs="X1"):
    """<obs>(t) starting from |+0>."""
    H = _H(J, h)
    evals, evecs = np.linalg.eigh(H)
    O = X1 if obs == "X1" else Z1
    out = np.empty(len(t_array), dtype=float)
    v = evecs.conj().T @ PLUS0
    for i, t in enumerate(t_array):
        psi_t = evecs @ (np.exp(-1j * evals * t) * v)
        out[i] = float((psi_t.conj() @ O @ psi_t).real)
    return out


def generate_traces(n_samples=50, seed=42, T=12.0, n_t=200):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, T, n_t)
    out = []
    for _ in range(n_samples):
        J = float(rng.uniform(0.2, 1.0))
        h = float(rng.uniform(0.2, 1.0))
        out.append({
            "t": t,
            "z1": trace(J, h, t, "Z1"),
            "x1": trace(J, h, t, "X1"),
            "J_true": J,
            "h_true": h,
        })
    return out


# ---------------------------------------------------------------- recovery
def _try_purediscover_fit(X, y, var_names):
    for mod_path, fn_name in [
        ("purediscover", "discover"),
        ("purediscover", "discover_expression"),
        ("purediscover.discovery", "discover"),
    ]:
        try:
            mod = __import__(mod_path, fromlist=[fn_name])
            fn = getattr(mod, fn_name, None)
            if fn is None:
                continue
            try:
                return fn(X, y, var_names=var_names)
            except TypeError:
                return fn(X, y)
        except (ImportError, Exception):
            continue
    return None


def recover(sample):
    """Recover (J, h) from (z1, x1) traces."""
    t, z1_obs, x1_obs = sample["t"], sample["z1"], sample["x1"]

    # Fast path: fit two-parameter (J, h) model to both traces
    def loss(p):
        J_, h_ = p
        if J_ <= 0 or h_ <= 0:
            return 1e10
        zp = trace(J_, h_, t, "Z1")
        xp = trace(J_, h_, t, "X1")
        return float(np.mean((zp - z1_obs) ** 2) + np.mean((xp - x1_obs) ** 2))

    # Coarse grid then Nelder-Mead
    best = (np.inf, 0.5, 0.5)
    for J0 in np.linspace(0.15, 1.1, 10):
        for h0 in np.linspace(0.15, 1.1, 10):
            l = loss((J0, h0))
            if l < best[0]:
                best = (l, J0, h0)

    from scipy.optimize import minimize
    res = minimize(loss, [best[1], best[2]], method="Nelder-Mead",
                   options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 2000})
    return {
        "kind": "parametric",
        "J_hat": float(res.x[0]),
        "h_hat": float(res.x[1]),
        "loss": float(res.fun),
    }
