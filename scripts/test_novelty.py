import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from purediscover_ext.novelty import detect_structured_residual


def _fit_harmonics(x, y, omegas):
    cols = []
    for w in omegas:
        cols.append(np.cos(w * x)); cols.append(np.sin(w * x))
    M = np.column_stack(cols)
    coefs, *_ = np.linalg.lstsq(M, y, rcond=None)
    return M @ coefs


def main():
    x = np.linspace(0, 10, 400)
    y = np.cos(2 * x) + 0.3 * np.sin(5 * x)

    print("=== Round 1: fit only cos(2x) ===")
    pred1 = _fit_harmonics(x, y, [2.0])
    resid1 = y - pred1
    rms1 = float(np.sqrt(np.mean(resid1 ** 2)))
    print(f"  fit RMS: {rms1:.4f}")

    sug = detect_structured_residual(resid1, x)
    if sug is None:
        print("  Detector: no structured residual (FAIL)"); return 1
    print(f"  Detector: structured residual found")
    print(f"    omega_fft = {sug['omega_fft']:.4f}")
    print(f"    omega_ref = {sug['omega']:.4f}  (expect 5.000)")
    print(f"    amp       = {sug['amp']:.4f}  (expect ~0.300)")
    print(f"    SNR       = {sug['snr']:.1f}x median")

    print("\n=== Round 2: fit cos(2x) + refined harmonic ===")
    pred2 = _fit_harmonics(x, y, [2.0, sug["omega"]])
    resid2 = y - pred2
    rms2 = float(np.sqrt(np.mean(resid2 ** 2)))
    print(f"  fit RMS: {rms2:.6f}")

    sug2 = detect_structured_residual(resid2, x)
    conv = sug2 is None
    print(f"  Detector round 2: {'converged (no structured residual)' if conv else 'still structured'}")

    ok_omega = abs(sug["omega"] - 5.0) < 1e-3
    ok_amp = abs(sug["amp"] - 0.30) < 0.01
    ok_converge = conv and rms2 < 1e-3
    ok = ok_omega and ok_amp and ok_converge
    print(f"\nPASS: {ok}  (omega={ok_omega}, amp={ok_amp}, converged={ok_converge})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
