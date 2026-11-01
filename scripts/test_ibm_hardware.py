#!/usr/bin/env python3
"""Run the ZZ Hamiltonian pipeline on real IBM Quantum hardware.

THIS SUBMITS REAL JOBS. Uses QPU time and queue.

HONEST SCOPE:
- NISQ test. Expect 15-40% parameter error, not simulator accuracy.
- Trotterized exp(-iHt): circuit depth ~20 gates. Gate errors dominate.
- 3 time points x 2 observables = 6 measurements for 2 parameters.
- A correct hardware run = sanity checks pass AND (J,h) within 50% of truth.
  High precision is NOT expected and would be suspicious.

What this proves: the pipeline submits jobs, retrieves counts, and fits
parameters from real qubit measurements.

What this does NOT prove: parameter recovery at simulator accuracy.
"""
from __future__ import annotations
import os, sys, time, argparse
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp
from qiskit.synthesis import LieTrotter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from purediscover_ext.qiskit_backend import exact_expectation

try:
    from qiskit_ibm_runtime import QiskitRuntimeService
    from qiskit_ibm_runtime import SamplerV2 as Sampler
except ImportError:
    from qiskit_ibm_runtime import QiskitRuntimeService, Sampler


def _make_sampler(backend):
    try:
        return Sampler(mode=backend)
    except TypeError:
        try:
            return Sampler(backend=backend)
        except TypeError:
            return Sampler(backend)


def _H_op(J, h):
    return SparsePauliOp.from_list([
        ("XX", J), ("YY", J), ("ZZ", J),
        ("ZI", h), ("IZ", h),
    ])


def build_evolution_circuit(J, h, t, obs_name, reps=2):
    qc = QuantumCircuit(2, 1)
    qc.h(1)
    evo = PauliEvolutionGate(_H_op(J, h), time=t, synthesis=LieTrotter(reps=reps))
    qc.append(evo, [0, 1])
    if obs_name == "X1":
        qc.h(1)
    qc.measure([1], [0])
    return qc


def build_sanity(prep):
    qc = QuantumCircuit(2, 1)
    if prep == "+0":
        qc.h(1)
    qc.measure([1], [0])
    return qc


def z_expectation(counts, shots):
    exp = 0.0
    for bits, c in counts.items():
        exp += (1 if bits[-1] == "0" else -1) * c
    return exp / shots


def get_counts(result, idx):
    data = result[idx].data
    for attr in ("c", "meas", "cr", "c0"):
        if hasattr(data, attr):
            return getattr(data, attr).get_counts()
    for v in data.__dict__.values():
        if hasattr(v, "get_counts"):
            return v.get_counts()
    raise RuntimeError("no classical register found in result")


def fit_J_h(z1_by_t, x1_by_t, times):
    from scipy.optimize import minimize
    t_arr = np.array(times)
    z1_obs = np.array([z1_by_t[t] for t in times])
    x1_obs = np.array([x1_by_t[t] for t in times])

    def loss(p):
        J_, h_ = p
        if not (0.05 <= J_ <= 1.5 and 0.05 <= h_ <= 1.5):
            return 1e10
        zp = exact_expectation(J_, h_, t_arr, "Z1")
        xp = exact_expectation(J_, h_, t_arr, "X1")
        return float(np.mean((zp - z1_obs) ** 2) + np.mean((xp - x1_obs) ** 2))

    best = (np.inf, 0.5, 0.5)
    for J0 in np.linspace(0.1, 1.3, 8):
        for h0 in np.linspace(0.1, 1.3, 8):
            l = loss((J0, h0))
            if l < best[0]:
                best = (l, J0, h0)
    res = minimize(loss, [best[1], best[2]], method="Nelder-Mead",
                   options={"xatol": 1e-6, "fatol": 1e-9, "maxiter": 1500})
    return float(res.x[0]), float(res.x[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="ibm_marrakesh")
    ap.add_argument("--shots", type=int, default=2048)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    J_true, h_true = 0.7, 0.4
    times = [0.5, 1.0, 1.5]

    circuits, labels = [], []
    circuits.append(build_sanity("00")); labels.append("sanity_00")
    circuits.append(build_sanity("+0")); labels.append("sanity_p0")
    for t in times:
        circuits.append(build_evolution_circuit(J_true, h_true, t, "Z1", reps=args.reps))
        labels.append(f"Z1_t{t}")
    for t in times:
        circuits.append(build_evolution_circuit(J_true, h_true, t, "X1", reps=args.reps))
        labels.append(f"X1_t{t}")

    print(f"Prepared {len(circuits)} circuits, shots={args.shots}, reps={args.reps}")
    if args.dry:
        print("DRY RUN - not submitting")
        for lbl, qc in zip(labels, circuits):
            print(f"  {lbl}: depth={qc.depth()} ops={dict(qc.count_ops())}")
        return 0

    print("\n*** SUBMITTING REAL HARDWARE JOBS ***")
    print("*** This uses QPU time and may queue for 5-30 min ***")

    service = QiskitRuntimeService(channel="ibm_quantum_platform")
    backend = service.backend(args.backend)
    st = backend.status()
    print(f"\nBackend: {backend.name}  qubits={backend.num_qubits}  pending={st.pending_jobs}")

    print("\nTranspiling...")
    t0 = time.time()
    transpiled = transpile(circuits, backend=backend, optimization_level=2)
    print(f"  transpile time: {time.time()-t0:.1f}s")
    print(f"  depths: {[t.depth() for t in transpiled]}")

    sampler = _make_sampler(backend)
    pubs = [(t,) for t in transpiled]

    print("\nSubmitting...")
    job = sampler.run(pubs, shots=args.shots)
    print(f"  job_id: {job.job_id()}")
    print(f"  status: {job.status()}")
    print("\nWaiting for completion (Ctrl-C to abort locally, job continues remotely)...")
    t0 = time.time()
    result = job.result()
    print(f"  done in {time.time()-t0:.1f}s")
    print(f"  final status: {job.status()}")
    try:
        print(f"  metrics: {job.metrics()}")
    except Exception:
        pass

    print("\n=== Raw counts ===")
    all_counts = []
    for i, lbl in enumerate(labels):
        c = get_counts(result, i)
        all_counts.append(c)
        print(f"  {lbl}: {c}")

    print("\n=== Sanity ===")
    z00 = z_expectation(all_counts[0], args.shots)
    zp0 = z_expectation(all_counts[1], args.shots)
    print(f"  <Z1>|00> = {z00:+.3f}  (expect +1.0)")
    print(f"  <Z1>|+0> = {zp0:+.3f}  (expect +0.5)")
    # |+0> measured in Z basis: <Z1> = 0, not 0.5. Hadamard on q1 gives
    # 50/50 counts; accept |<Z1>| < 0.1 (5 sigma of 2048 shots).
    sanity_ok = (z00 > 0.85) and (abs(zp0) < 0.10)
    print(f"  sanity PASS: {sanity_ok}")

    print("\n=== Evolution traces ===")
    z1_by_t, x1_by_t = {}, {}
    for i, t in enumerate(times):
        z1_by_t[t] = z_expectation(all_counts[2 + i], args.shots)
        x1_by_t[t] = z_expectation(all_counts[5 + i], args.shots)
        z1_sim = exact_expectation(J_true, h_true, np.array([t]), "Z1")[0]
        x1_sim = exact_expectation(J_true, h_true, np.array([t]), "X1")[0]
        print(f"  t={t:.2f}  Z1={z1_by_t[t]:+.3f} (sim {z1_sim:+.3f})  "
              f"X1={x1_by_t[t]:+.3f} (sim {x1_sim:+.3f})")

    print("\n=== Fitting (J, h) from measured 6 points ===")
    J_hat, h_hat = fit_J_h(z1_by_t, x1_by_t, times)
    j_err = abs(J_hat - J_true) / J_true
    h_err = abs(h_hat - h_true) / h_true
    print(f"  true:   J={J_true:.4f}  h={h_true:.4f}")
    print(f"  fitted: J={J_hat:.4f}  h={h_hat:.4f}")
    print(f"  rel err: J={j_err*100:.1f}%  h={h_err*100:.1f}%")

    print("\n=== Honest summary ===")
    print("  NISQ hardware: 15-40% error is normal. High precision is NOT expected.")
    print("  Correctness bar: sanity passes AND both params within 50% of truth.")
    ok = sanity_ok and j_err < 0.5 and h_err < 0.5
    print(f"  HARDWARE PASS: {ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
