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

### MDL scorer structural-vs-residual weighting
**Fixed (v2).** The legacy `mdl_score` (now `_mdl_score_v1`) used
`sum(log2(1 + r/scale))` for data bits, which grows sublinearly in
residual improvement — a 100x fit gain buys ~1 bit for n=100, while one
extra structural node costs ~40 bits. Effectively demands ~3000x fit
improvement to justify one node; the near-optimal-MAE filter masked this.

v2 uses the standard two-part Gaussian universal code:
`L = n_nodes*log2(lib) + n_const*log2(range) + (n/2)*log2(2*pi*e*RSS/n)`,
reported as gain vs a median-baseline null model (negative = compresses
better than flat). Ranking is preserved on all tested cases: linear, cosine,
damped Rabi, Hamiltonian (0.05% PASS), Keeling, recursive-additive.
`PUREDISCOVER_MDL=v1` retains the legacy scorer for reproducibility.

### Previously out-of-scope gaps now closed-restricted

Two gaps previously listed here as "out of scope, not attempted" were closed
in the follow-up session. Both carry honest restricted scope; see the
corresponding sections below.

- **Implicit laws** — closed-restricted (fixed-point iteration on Colebrook).
  See `## Gap: Implicit laws (fixed-point iteration)` below.
- **Design real unknown experiments** — closed-restricted (unknown-parameter
  recovery on a random Hamiltonian family, with joint ⟨Z₁⟩, ⟨X₁⟩ measurement).
  See `## Gap: Unknown-Hamiltonian discovery` below.

---

## Known limitations within demonstrated cases

- Short windows (< 1 cycle of a component) cannot resolve that component's
  frequency. Verified: 1D additive fails at t∈[0.1,6.0], passes at t∈[0.1,20.0].
- Keeling CO₂ trend is nearly linear over 10-year holdout; test MAE
  is dominated by the acceleration in the last decade, not by the annual
  cycle fit.
- All quantum results use a simulated oracle. Real quantum hardware not used.
- The original Hamiltonian-learning test (|01⟩ initial state, ⟨Z₁⟩
  observable) still only recovers `J`, because `h` decouples from ⟨Z₁⟩ in
  that block. Extending to recover `h` requires the pair (⟨Z₁⟩, ⟨X₁⟩) with
  |+0⟩ initial state; that is what `unknown_hamiltonian.py` now does (see
  closed-restricted section below). The original test is unchanged.

---

## Reproduce

    python scripts/test_hamiltonian_learning.py    # headline: J to 0.05%
    python scripts/real_data_keeling_v2.py         # headline: annual cycle
    python scripts/test_4d.py                      # 4D discovery
    python scripts/test_recursive_additive_1d.py   # 1D multi-frequency
    python scripts/test_recursive_damped_rabi.py   # damped Rabi
    python scripts/test_3d.py                      # 3D discovery
    python scripts/run_full_loop_2d.py             # full autonomy loop

## Gap: Implicit laws (fixed-point iteration)

Status: ✅ Closed (restricted).

The pipeline is given an iteration structure `y_{n+1} = g(x, y_n)` and must
recover `g`. Demonstrated on Colebrook's equation for turbulent pipe friction.
Discovery path is reported honestly: symbolic if purediscover's regressor
exposes an entry point, otherwise log-linear structure + coefficients via
nonlinear least squares. Convergence to the true fixed point is verified on
200 held-out (Re, ks_D) pairs.

What is proven: the pipeline handles implicit relations given an iteration
structure, and the composed fixed point matches Colebrook to < 1e-3 relative
error.

What is not proven: fully autonomous implicit-law discovery from scratch.

## Gap: Unknown-Hamiltonian discovery

Status: ✅ Closed (restricted).

Tested on H = J(XX+YY+ZZ) + h(ZI+IZ) with random (J, h) in [0.2, 1.0]^2.
Note: with initial state |+0>, the observable <Z1>(t) = 1/2 - 1/2*cos(2Jt)
depends only on J, contrary to a first reading of the Hamiltonian's symmetry.
To recover both parameters we measure the pair (<Z1>, <X1>) where
<X1>(t) = 1/2[cos(2h t) + cos(2(h-J) t)]. Recovery is verified against oracle
(J, h) across 50 random systems with a < 5% relative error criterion on both.

What is proven: (J, h) recovered without being told the model, from
unannounced random parameters.

What is not proven: novel-physics discovery. The pipeline still has cos(...)
in its library; this is unknown-parameter recovery in a known model family.

---

### Identifiability note (Colebrook fixed-point)

The log-linear form `a·log10(b·ks + c/(Re·√f)) + d` has a scale degeneracy:
only the ratio `c/b` is identifiable when `d` is free. With `d` unconstrained,
the first fit returned `(a=-2, b=0.579, c=5.377, d=0.662)` — still predicting
the fixed point to 1e-15, but not the canonical parameters. Constraining
`d = 0` breaks the degeneracy and recovers `(a, b, c) = (-2, 1/3.7, 2.51)`
uniquely; the held-out fixed point then matches Colebrook to 7e-15 relative
error across 200 random `(Re, ks_D)` pairs.

General lesson for implicit-law discovery: perfect prediction at the fixed
point does not imply parameter identification. Any test reporting recovered
coefficients must either (a) constrain the gauge or (b) report only
gauge-invariant combinations (here, `c/b = 9.287`).
