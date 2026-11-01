import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from purediscover_ext.active_loop import active_recovery


def main():
    J_true, h_true = 0.7, 0.4
    t_init = [0.1, 0.2, 0.3, 0.4, 0.5]   # short times; poorly conditioned

    history = active_recovery(J_true, h_true, "X1", t_init, n_iter=3, n_new=5)

    print(f"=== Active experiment loop (X1)   truth: J={J_true}, h={h_true} ===")
    print(f"{'iter':>5} {'n':>4} {'J_hat':>8} {'h_hat':>8} {'J err':>10} {'h err':>10}")
    print("-" * 55)
    for r in history:
        print(f"{r['iter']:>5} {r['n']:>4} {r['J_hat']:>8.4f} {r['h_hat']:>8.4f} {r['j_err']:>10.2e} {r['h_err']:>10.2e}")

    err0 = max(history[0]["j_err"], history[0]["h_err"])
    errf = max(history[-1]["j_err"], history[-1]["h_err"])
    factor = err0 / errf if errf > 0 else float("inf")

    print(f"\nInitial max err: {err0:.3e}")
    print(f"Final   max err: {errf:.3e}")
    print(f"Improvement factor: {factor:.1f}x")

    ok = factor >= 10.0
    print(f"\nPASS (>=10x improvement): {ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
