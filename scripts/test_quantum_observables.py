import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from purediscover_ext.quantum_observables import (
    SINGLE, CORRELATORS_2Q, BELL, PARITY, ALL, rank_observables,
)


def main():
    print(f"Quantum observable library")
    print(f"  single-qubit Paulis: {len(SINGLE)}: {list(SINGLE)}")
    print(f"  2-qubit correlators: {len(CORRELATORS_2Q)}: {list(CORRELATORS_2Q)}")
    print(f"  Bell projectors: {len(BELL)}: {list(BELL)}")
    print(f"  parity operators: {len(PARITY)}: {list(PARITY)}")
    print(f"  total: {len(ALL)}")

    print(f"\n=== Ranking for ZZ Hamiltonian, J=0.7 h=0.4 ===")
    scores = rank_observables()
    print(f"{'observable':<12} {'rank':>4} {'cond':>12}")
    for name, rank, cond in scores:
        cond_s = f"{cond:.2e}" if cond < 1e100 else "inf"
        print(f"{name:<12} {rank:>4} {cond_s:>12}")

    # Sanity checks
    z1_rank = next(r for n, r, c in scores if n == "Z1")
    x1_rank = next(r for n, r, c in scores if n == "X1")
    xx_rank = next(r for n, r, c in scores if n == "XX")

    print(f"\n=== Sanity ===")
    print(f"  Z1 insufficient (h invisible): rank={z1_rank}, expect 1")
    print(f"  X1 sufficient:                rank={x1_rank}, expect 2")
    print(f"  XX correlator:                rank={xx_rank}, expect 1 (XX commutes with both J and h terms sym)")

    ok = (z1_rank == 1 and x1_rank == 2 and len(ALL) >= 15)
    print(f"\nPASS: {ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
