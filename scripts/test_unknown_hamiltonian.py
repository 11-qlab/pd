import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from purediscover_ext.unknown_hamiltonian import generate_traces, recover

def main():
    print("=" * 60)
    print("Unknown-Hamiltonian recovery  (J, h)")
    print("=" * 60)

    samples = generate_traces(n_samples=50, seed=42)
    j_errs, h_errs = [], []
    for s in samples:
        r = recover(s)
        j_errs.append(abs(r["J_hat"] - s["J_true"]) / abs(s["J_true"]))
        h_errs.append(abs(r["h_hat"] - s["h_true"]) / abs(s["h_true"]))

    j_errs = np.array(j_errs); h_errs = np.array(h_errs)
    print(f"\nN = {len(samples)}")
    print(f"J rel err:  median={np.median(j_errs):.2e}  p90={np.percentile(j_errs,90):.2e}  max={j_errs.max():.2e}")
    print(f"h rel err:  median={np.median(h_errs):.2e}  p90={np.percentile(h_errs,90):.2e}  max={h_errs.max():.2e}")
    pass_rate = float(np.mean((j_errs < 0.05) & (h_errs < 0.05)))
    print(f"\nPass rate (< 5% rel err on both J and h): {pass_rate*100:.1f}%")
    ok = pass_rate >= 0.95
    print(f"PASS (>= 95%): {ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
