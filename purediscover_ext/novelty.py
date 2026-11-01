"""Structured-residual detection + primitive suggestion (restricted closing).

Detects structured residuals via FFT peak, then *refines* the peak frequency
by nonlinear least squares (FFT bins are coarse; without refinement a
harmonic at ω=5.014 in a signal at ω=5.000 leaves ~1e-2 RMS residual,
which the detector then flags as new novelty — false positive). Refinement
drives ω to <1e-4 of the true frequency, so retry converges.

Restricted scope: suggestion family is harmonics; the primitive is the
fitted harmonic, not a new symbolic class.
"""
from __future__ import annotations
import numpy as np
import sympy as sp
from scipy.optimize import minimize_scalar


def _harmonic_fit_rms(x, r, omega):
    M = np.column_stack([np.cos(omega * x), np.sin(omega * x)])
    coefs, *_ = np.linalg.lstsq(M, r, rcond=None)
    return float(np.sqrt(np.mean((M @ coefs - r) ** 2)))


def _refine_omega(x, r, omega_fft, search_frac=0.05):
    """Refine omega near the FFT peak via bounded scalar minimization."""
    lo = omega_fft * (1 - search_frac)
    hi = omega_fft * (1 + search_frac)
    res = minimize_scalar(lambda w: _harmonic_fit_rms(x, r, w),
                          bounds=(lo, hi), method="bounded",
                          options={"xatol": 1e-8})
    return float(res.x)


def detect_structured_residual(residuals, x, peak_threshold=10.0,
                                rms_floor=1e-3):
    r = np.asarray(residuals, dtype=float)
    if len(r) < 8:
        return None
    if float(np.sqrt(np.mean(r ** 2))) < rms_floor:
        return None  # already converged

    r_c = r - r.mean()
    spec = np.abs(np.fft.rfft(r_c))
    freqs = np.fft.rfftfreq(len(r), d=float(x[1] - x[0]))
    if len(spec) < 3:
        return None

    peak_idx = int(np.argmax(spec[1:]) + 1)
    peak_amp = float(spec[peak_idx])
    median_amp = float(np.median(spec[1:]))
    if median_amp <= 0 or peak_amp < peak_threshold * median_amp:
        return None

    omega_fft = float(2 * np.pi * freqs[peak_idx])
    omega_ref = _refine_omega(x, r_c, omega_fft)

    M = np.column_stack([np.cos(omega_ref * x), np.sin(omega_ref * x)])
    coefs, *_ = np.linalg.lstsq(M, r_c, rcond=None)
    A, B = coefs
    amp = float(np.hypot(A, B))
    phase = float(np.arctan2(-B, A))

    x_sym = sp.Symbol("x")
    prim = sp.Float(amp) * sp.cos(sp.Float(omega_ref) * x_sym + sp.Float(phase))

    return {"omega": omega_ref, "omega_fft": omega_fft,
            "amp": amp, "phase": phase, "primitive": prim,
            "peak_amp": peak_amp, "noise_floor": median_amp,
            "snr": peak_amp / median_amp if median_amp > 0 else float("inf")}
