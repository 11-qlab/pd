import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from purediscover_ext.solvers import (
    discover_implicit_law, predict_with_implicit, newton_solve, brentq_solve,
)


def colebrook_oracle(Re, ks, f0=0.02, tol=1e-10, max_iter=50):
    f = f0
    for _ in range(max_iter):
        f_new = 1.0 / (-2.0 * np.log10(ks/3.7 + 2.51/(Re*np.sqrt(f)))) ** 2
        if abs(f_new - f) < tol:
            return f_new
        f = f_new
    return f


def main():
    rng = np.random.default_rng(0)
    n = 100
    Re = 10 ** rng.uniform(4, 6, n)
    ks = 10 ** rng.uniform(-5, -2, n)
    f_true = np.array([colebrook_oracle(Re[i], ks[i]) for i in range(n)])
    data = [{"x": (Re[i], ks[i]), "y": float(f_true[i])} for i in range(n)]

    print("=== Implicit-law discovery (Colebrook) ===")
    result = discover_implicit_law(data)
    for r in result["valid"]:
        print(f"  {r['template']:<10s}  BIC={r['bic']:+8.2f}  rss={r['rss']:.3e}")
    if result["rejected"]:
        print(f"  rejected (degenerate): {result['rejected']}")
    if not result["valid"]:
        print("  no valid template — FAIL")
        return 1

    best = result["valid"][0]
    print(f"\nChosen: {best['template']}")
    print(f"  params: {['%.4g' % p for p in best['params']]}")

    n_t = 50
    Re_t = 10 ** rng.uniform(4, 6, n_t)
    ks_t = 10 ** rng.uniform(-5, -2, n_t)
    f_oracle = np.array([colebrook_oracle(Re_t[i], ks_t[i]) for i in range(n_t)])
    f_pred = np.array([predict_with_implicit(best, Re_t[i], ks_t[i]) for i in range(n_t)])
    mask = np.isfinite(f_pred)
    n_valid = int(mask.sum())

    print(f"\nHeld-out via brentq on F=0:")
    print(f"  valid: {n_valid}/{n_t}")
    if n_valid == 0:
        print("  no valid predictions — FAIL")
        return 1
    rel = np.abs(f_pred[mask] - f_oracle[mask]) / f_oracle[mask]
    print(f"  max rel err:  {rel.max():.3e}")
    print(f"  mean rel err: {rel.mean():.3e}")

    print(f"\nSolver smoke tests:")
    print(f"  newton(x^2-4, y0=3)    = {newton_solve(lambda y: y*y-4, 3.0):.6f}  (expect 2)")
    print(f"  brentq(x^2-4, [0,5])   = {brentq_solve(lambda y: y*y-4, 0.0, 5.0):.6f}  (expect 2)")

    ok = rel.max() < 1e-2 and n_valid >= n_t * 0.9
    print(f"\nPASS (max rel err < 1e-2, >=90% valid): {ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
