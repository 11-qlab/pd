import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from purediscover_ext.observable_selection import score_observables


def main():
    t = np.linspace(0, 10, 200)
    J, h = 0.7, 0.4

    candidates = [
        ("Z1",), ("Z2",), ("X1",), ("Y1",), ("X2",), ("Y2",),
        ("Z1", "Z2"), ("Z1", "X1"), ("X1", "Y1"), ("Z1", "Z2", "X1"),
    ]
    scores = score_observables(candidates, J, h, t)

    print("=== Observable ranking (lower cond = better; rank<2 = insufficient) ===")
    print(f"{'observables':<22} {'rank':>6} {'cond':>12}")
    print("-" * 45)
    for s in scores:
        cond_s = f"{s['cond']:.2e}" if np.isfinite(s["cond"]) else "inf"
        print(f"{str(s['obs']):<22} {s['rank']:>6} {cond_s:>12}")

    z1 = next(s for s in scores if s["obs"] == ("Z1",))
    x1 = next(s for s in scores if s["obs"] == ("X1",))

    ok_z1 = z1["rank"] < 2
    ok_x1 = x1["rank"] >= 2 and x1["cond"] < 1e4

    print(f"\nZ1 alone insufficient (rank<2): {ok_z1}")
    print(f"X1 alone sufficient (rank>=2, cond<1e4): {ok_x1}")
    print(f"\nPASS: {ok_z1 and ok_x1}")
    return 0 if (ok_z1 and ok_x1) else 1


if __name__ == "__main__":
    sys.exit(main())
