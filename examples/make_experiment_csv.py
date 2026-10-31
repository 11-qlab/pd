"""Write two small experiment CSVs with realistic noise.

    experiment.csv          Ohm's law, V = I*R    -> KNOWN (ohm)
    experiment_novel.csv    radioactive decay,   -> NOVEL
                            N = exp(-lambda * t)

Both have 40 rows and 1% multiplicative noise (typical bench
measurement precision). Run this once; edit if you want a different
scenario.
"""
import csv
import random

rng = random.Random(42)


def _noisy(y: float, noise: float = 0.01) -> float:
    return y * (1.0 + rng.gauss(0.0, noise))


# --- Ohm's law ---------------------------------------------------------
with open("experiment.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["I", "R", "V"])
    for _ in range(40):
        I = rng.uniform(0.1, 5.0)     # 100 mA .. 5 A
        R = rng.uniform(1.0, 100.0)   # 1 ohm .. 100 ohm
        V = _noisy(I * R)
        w.writerow([f"{I:.4f}", f"{R:.4f}", f"{V:.4f}"])
print("wrote experiment.csv         (Ohm's law, V = I*R)")

# --- Radioactive decay -------------------------------------------------
with open("experiment_novel.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["t", "lam", "N"])
    for _ in range(40):
        t = rng.uniform(0.2, 3.0)     # 0.2 .. 3 half-lives-ish
        lam = rng.uniform(0.5, 2.0)   # decay constant
        N = _noisy(2.718281828 ** (-lam * t))   # exp(-lam*t), N0=1
        w.writerow([f"{t:.4f}", f"{lam:.4f}", f"{N:.6f}"])
print("wrote experiment_novel.csv   (decay, N = exp(-lam*t))")
