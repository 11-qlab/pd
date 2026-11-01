"""Generate quantum microgravity CSVs — 2 columns only (var, y).

Prior version had 3 columns (var, display_name, y) which caused
batch_discover to trivially fit y = display_name. Fixed.
"""
import csv
import math
import pathlib
import random

OUT = pathlib.Path("datasets/input")
OUT.mkdir(parents=True, exist_ok=True)


def make_phase():
    rng = random.Random(11)
    rows = []
    for _ in range(30):
        T = rng.uniform(0.1, 2.7)
        phi = 4.2 * T**2 + 0.08 * T**3
        # 0.5% noise so the T^3 term (0.08) is above the noise floor
        # on the T^2 term (4.2 * 0.005 = 0.021 per sample)
        phi *= (1.0 + rng.gauss(0.0, 0.005))
        rows.append((T, phi))
    path = OUT / "microgravity_mz_phase.csv"
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["T_s", "y"])
        for T, phi in rows:
            w.writerow([f"{T:.6f}", f"{phi:.6f}"])
    print(f"wrote {path}  ({len(rows)} rows)  2 cols: T_s, y")


def make_bec_tof():
    rng = random.Random(22)
    rows = []
    for _ in range(30):
        t = rng.uniform(10.0, 300.0)
        sigma = 12.0 * math.sqrt(1.0 + (t / 85.0)**2)
        sigma *= (1.0 + rng.gauss(0.0, 0.02))
        rows.append((t, sigma))
    path = OUT / "bec_tof_expansion.csv"
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_ms", "y"])
        for t, sigma in rows:
            w.writerow([f"{t:.4f}", f"{sigma:.4f}"])
    print(f"wrote {path}  ({len(rows)} rows)  2 cols: t_ms, y")


def make_fringe_spacing():
    rng = random.Random(33)
    rate = 0.41
    anchors = [(180.0, 75.0), (260.0, 107.0)]
    rows = list(anchors)
    for _ in range(28):
        T = rng.uniform(60.0, 300.0)
        lam = rate * T * (1.0 + rng.gauss(0.0, 0.02))
        rows.append((T, lam))
    rows.sort()
    path = OUT / "fringe_spacing.csv"
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["T_ex_ms", "y"])
        for T, lam in rows:
            w.writerow([f"{T:.4f}", f"{lam:.4f}"])
    print(f"wrote {path}  ({len(rows)} rows)  2 cols: T_ex_ms, y")


if __name__ == "__main__":
    make_phase()
    make_bec_tof()
    make_fringe_spacing()
    print("\nAll 3 CSVs regenerated as 2-column (var, y)")
