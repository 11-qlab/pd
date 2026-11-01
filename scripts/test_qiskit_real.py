import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from purediscover_ext.qiskit_backend import (
    exact_expectation, shots_expectation, fit_J_h_from_traces,
)
from purediscover_ext.observable_selection import evolve_expectation as np_evolve


def main():
    J, h = 0.7, 0.4
    t = np.linspace(0, 10, 30)

    print("=== Tier 1: exact statevector vs numpy ===")
    z1_np = np_evolve(J, h, t, "Z1")
    z1_qk = exact_expectation(J, h, t, "Z1")
    err = float(np.max(np.abs(z1_np - z1_qk)))
    print("  Z1 max diff: %.3e" % err)
    ok1 = err < 1e-10
    print("  PASS (<1e-10): %s" % ok1)

    print("")
    print("=== Tier 2: shot noise (1024 shots) ===")
    z1_shots = shots_expectation(J, h, t, "Z1", shots=1024, seed=42)
    err_shots = float(np.max(np.abs(z1_np - z1_shots)))
    sigma = 1.0 / np.sqrt(1024)
    print("  Z1 max diff: %.4f  (1-sigma ~ %.3f)" % (err_shots, sigma))
    print("  Z1 max diff / sigma: %.2f" % (err_shots / sigma))
    ok2 = err_shots < 5 * sigma
    print("  PASS (<5 sigma): %s" % ok2)

    print("")
    print("=== Tier 3: (J,h) recovery from shots only ===")
    t_z = t[::2]
    t_x = t[1::2]
    z1_obs = shots_expectation(J, h, t_z, "Z1", shots=4096, seed=1)
    x1_obs = shots_expectation(J, h, t_x, "X1", shots=4096, seed=2)
    J_hat, h_hat = fit_J_h_from_traces(0.5, 0.5, t_z, z1_obs, t_x, x1_obs)
    j_err = abs(J_hat - J) / J
    h_err = abs(h_hat - h) / h
    print("  true:    J=%.4f h=%.4f" % (J, h))
    print("  fitted:  J=%.4f h=%.4f" % (J_hat, h_hat))
    print("  rel err: J=%.3e h=%.3e" % (j_err, h_err))
    ok3 = j_err < 0.10 and h_err < 0.10
    print("  PASS (<10%% each): %s" % ok3)

    all_ok = ok1 and ok2 and ok3
    print("")
    print("=== PASS overall: %s ===" % all_ok)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
