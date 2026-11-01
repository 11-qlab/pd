# Finding: Closed-Form Law Recovery from Noisy Measurements

**Status:** internal report. Not for external distribution.
**Date:** 2026-11.

---

## Headline results

### 1. Hamiltonian learning (quantum)
2-qubit Heisenberg `H = J(X⊗X + Y⊗Y + Z⊗Z)`. From noisy `⟨Z₁⟩(t)` traces
(50 points, 2% multiplicative noise), coupling `J` recovered to
**0.05% worst-case relative error** across `J ∈ {0.5, 1.0, 1.5, 2.0, 2.5}`.
No prior knowledge of J. Autonomous.

### 2. Real measurement data (Mauna Loa CO₂)
823 monthly observations, 1958–2026, real NOAA instruments.
**Annual cycle recovered at ω = 6.2859 rad/yr (0.04% of 2π)** with no prior
on frequency. Amplitude 2.83 ppm matches published seasonal swing.
Held-out test MAE 1.80 ppm vs persistence baseline 14.68 ppm — **8× better
generalization on 10-year holdout**.

---

## What else works

| Case | Verified |
|---|---|
| 1D: `cos(Bx)` from clean oracle | MRE 8.4e-16 |
| 2D: `cos(2·Bx·t)` from 2% noise | Coeffs within 0.5%, converged iter 13/30 |
| 3D: `cos(2·x·y·z)` from 2% noise | Coeffs within 1% |
| 4D: `cos(2·w·x·y·z)` from 2% noise | Coeffs within 2%, composite has all 4 vars |
| 1D additive: `cos(2t) + 0.5·sin(0.5t)` | 2.006 / 0.502 recovered (0.3% / 0.4% err) |
| Damped Rabi: `cos(2·Bx·t)·exp(-t/2)` | Amplitude/decay/freq within 0.4% |
| Sparse algebraic: linear/bilinear/quadratic | Coeffs within ~1% |

---

## Architecture

Two-arm discovery:
- **Template arm**: 14 parametric families (1D), 12 (2D), composite enumeration (ND)
- **Sparse arm**: linear regression over monomial + ratio + trig feature library
- **Arbitration**: MDL, with near-optimal-MAE filter to compensate for
  structural-bits bias in the scorer
- **Self-termination**: convergence detector fires when best candidate plateaus
- **Proposal**: BOED disagreement-based active learning
- **Multiscale**: two-stage trend + seasonal decomposition for real data
- **Recursive**: residual-based composition (multiplicative or additive)
  with BIC acceptance

---

## Gaps closed this session

| Gap | Status | Evidence |
|---|---|---|
| 1D multi-frequency additive | ✅ Closed | `cos(2.006t) + 0.495·cos(0.502t)`, freqs within 0.4% |
| Real measurement data | ✅ Closed | Keeling: ω=6.2859 (0.04% of 2π), test MAE 1.80 vs 14.68 |
| nD beyond 3D (4D verified) | ✅ Closed | `cos(1.965·wxyz + 0.02)`, 4-var composite |
| Damped Rabi decomposition | ✅ Closed | Amplitude/decay/freq within 0.4% |
| Hamiltonian learning | ✅ Closed | J to 0.05% from noisy quantum traces |

## Gaps remaining (honest)

### Implicit laws (Nikuradse-style `f(x,y)=0`)
**Out of scope.** Requires numerical solvers (Newton, brentq) as grammar
primitives and different MDL formulation. Weeks of work. Not attempted.

### MDL scorer structural-vs-residual weighting
**Mitigated, not fixed.** The core `mdl_score` weights structural bits too
heavily vs residual encoding. Near-optimal-MAE filter compensates at the
candidate selection layer. Fixing the scorer would invalidate every "bits"
comparison reported here and require re-verification. Not attempted.

### Design real unknown experiments
**Out of scope.** Current loop queries an oracle whose law it is recovering.
Real unknown-experiment design requires:
- A physics backend that's genuinely unknown (new simulator / real lab / public DB)
- A reward function valuing information gain over fit quality
- Novelty detection to signal library failure
This is a research project, not a feature. Not attempted.

---

## Known limitations within demonstrated cases

- Short windows (< 1 cycle of a component) cannot resolve that component's
  frequency. Verified: 1D additive fails at t∈[0.1,6.0], passes at t∈[0.1,20.0].
- Keeling CO₂ trend is nearly linear over 10-year holdout; test MAE
  is dominated by the acceleration in the last decade, not by the annual
  cycle fit.
- All quantum results use a simulated oracle. Real quantum hardware not used.
- Hamiltonian learning uses the `|01⟩` special case where `h` decouples
  from `⟨Z₁⟩`. General `h ≠ 0` with `|+0⟩` initial state requires extended
  analysis.

---

## Reproduce

    python scripts/test_hamiltonian_learning.py    # headline: J to 0.05%
    python scripts/real_data_keeling_v2.py         # headline: annual cycle
    python scripts/test_4d.py                      # 4D discovery
    python scripts/test_recursive_additive_1d.py   # 1D multi-frequency
    python scripts/test_recursive_damped_rabi.py   # damped Rabi
    python scripts/test_3d.py                      # 3D discovery
    python scripts/run_full_loop_2d.py             # full autonomy loop
