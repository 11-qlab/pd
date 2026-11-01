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

---

## Quantum-specific capabilities (post-session)

The four restricted closings split along quantum/classical lines:

| Module | Test case | Domain | What's quantum-relevant |
|---|---|---|---|
| `solvers.py` | Colebrook friction | Classical | None (generic implicit-form fitter) |
| `observable_selection.py` | ZZ Hamiltonian | **Quantum** | Ranks measurement observables by Jacobian cond; auto-rejects ⟨Z₁⟩ alone (h invisible) |
| `novelty.py` | Harmonic detection | Generic | Applies to quantum spectra / time traces |
| `active_loop.py` | ZZ Hamiltonian | **Quantum** | D-optimal measurement-time selection for a quantum system |

### What the quantum modules prove

- **Observable selection**: given candidates {⟨Z₁⟩, ⟨X₁⟩, ⟨X₂⟩, ⟨Y₁⟩, ⟨Y₂⟩, ⟨Z₂⟩},
  the pipeline ranks ⟨X₁⟩ highest (cond=2.53) and ⟨Z₁⟩ insufficient (cond=9.4e10).
  This catches the same failure that broke the *original* unknown-Hamiltonian
  proposal — the human-supplied `|+0⟩ / ⟨Z₁⟩` combination cannot recover h.
  The pipeline now self-diagnoses this.

- **Active loop**: from 5 short-time initial points with 2% noise, D-optimal
  time selection recovers J and h to <5% after 3 rounds of 5 additional
  measurements each. Warm-start is essential: without it, D-optimal times
  land the refit in a different aliased basin of the periodic ZZ model.

### What remains open (quantum side)

- Joint design across observables AND times (currently sequential).
- Hardware constraints on measurable times / observables.
- Adaptive oracle selection (which backend to query next).
- Generalization beyond 2-qubit Pauli systems.

These are research-scope; the two closed items each prove a scoped capability.

### Identifiability note (implicit Colebrook)

The Colebrook template `a/√f + b*log10(ks/c + d/(Re√f)) + e` has a scale
gauge: `(b, c) → (λb, λc)` leaves F unchanged up to the log constant, and
`(a, b, e)` have one more residual degree of freedom when `F≈0` on the data.
Only `c` and `b/a` are identified by the data; individual `a`, `b` values
are not canonical. Perturbed-init recovery confirms this: from `c_init=1.5`,
the fit returns `(a,b,c)=(0.83, 1.65, 3.73)` with `b/a=2.000` and `c=3.73`;
held-out brentq predictions match the Colebrook oracle to 4e-10 regardless.
This is the same lesson as the fixed-point module: predicting y at new x
does not require identifying the gauge; only coefficients that enter the
observable are identifiable.


---

## Expanded templates / libraries / candidate sets

The previous restricted closings relied on minimal frames (2 templates,
12 primitives, 6 observables). Expanded to real-physics coverage:

| Module | Content |
|---|---|
| `physics_templates.py` | 14 laws across fluid dynamics, thermo, adsorption, enzyme, electrochem |
| `grammar_library.py` | 33 primitives with MDL priors; common ops 2.74 bits/node, rare 8.97 bits/node |
| `quantum_observables.py` | 21 candidates: single Paulis, 2q correlators, Bell projectors, parity |

### What this widens

The pipeline can fit or rank correctly across a wider physics surface without
the human re-supplying a template each time. Colebrook, Langmuir,
Michaelis-Menten, Butler-Volmer, and Bell-state observables are now available.

### What this does NOT prove

The frame is still human-supplied. If a law is not in the template library,
the pipeline fits the closest wrong template and returns a plausible-looking
answer. `check_all_templates_fail()` is the guard: it flags templates whose
best fit has RSS > 5% of data variance, signalling the true law may be
outside the library, but this is a heuristic, not proof of novelty.

Also, the information floor: with a 14-template library and a 21-observable
set, more data is required to rank correctly. Expanding the library helps
only when you have the data to justify the expanded search.

### A specific autonomy win (quantum)

The XX correlator is rejected for the ZZ Hamiltonian with cond=inf, not
because of numerical noise but because XX commutes with every term of H:
<XX>(t) is constant, carrying zero information about J or h. The pipeline
autonomously recognizes a symmetry-charge observable as informationally
null. Z1 is a weaker case: <Z1>(t) depends on J but not h, so it is
rank-deficient rather than null. Both calls - do not measure a conserved
quantity to learn the Hamiltonian - are what a physicist would make; the
pipeline now makes them automatically from the Jacobian.

### What remains research-scope

- Automatic template invention (novelty.py handles harmonics only)
- Automatic observable invention (candidates are human-supplied)
- Prior estimation from data (priors are human-assigned)


---

## Testing at multiple fidelity tiers (quantum)

`scripts/test_qiskit_real.py` runs three tiers against the ZZ Hamiltonian:

| Tier | Setup | Result |
|---|---|---|
| 1 | Qiskit statevector (exact) | Z1 matches numpy to 0.0 (identical) |
| 2 | Aer + 1024 shots (Born-rule sampling) | max diff 0.054, 1.74σ |
| 3 | Recover (J,h) from 4096-shot expectation traces | J, h to 2e-4 rel err |

Tier 3 accuracy is not a hardware claim: Aer-with-shots has zero decoherence,
only binomial measurement noise. Real NISQ adds T1/T2 and gate errors; expect
15-30% parameter recovery error on hardware, not 2e-4.

### A bug that only the shot tier exposed

Tier 1 (statevector) passed with zero error while Tier 2 was 32σ off — a
systematic, not noise. Root cause: Qiskit measures qubit 0 (LSB) but numpy's
"Z1" observable is Z on the first kron factor (= qubit 1). The statevector
check never exercises the measurement circuit, so the mismatch was invisible
until shot-based sampling was used. Fixed by measuring qubit 1 for Z1/X1.
General lesson: exact-evolution tests can verify a unitary's math and miss
its measurement wiring entirely; only tier-2 (shots) exposes it.


---

## Real IBM Quantum hardware — two backends

`scripts/test_ibm_hardware.py` submitted the same 8-circuit suite to two
physical IBM devices. Same pipeline, same circuits, independent hardware.

| | ibm_marrakesh | ibm_fez |
|---|---|---|
| Job ID | dbjmhlhffo6c73b9u9bg | dbjndnd6p51s73cl4dh0 |
| Queue | 52 min | 17 s |
| QPU | 7 s | 6 s |
| J_hat | 0.7019 (0.3% err) | 0.7082 (1.2% err) |
| h_hat | 0.3454 (13.6% err) | 0.3489 (12.8% err) |
| sanity_00 counts | {0: 2048} | {0: 2032, 1: 16} |
| sanity_p0 counts | {1: 1038, 0: 1010} | {1: 1043, 0: 1005} |

### What the two-run consistency shows

- J is recovered to ~1% on both devices. Independent hardware, independent
  calibrations, consistent answer.
- h is recovered ~13% low on both. This is systematic, not shot noise:
  the Z1 trace is depressed on hardware (Trotter + gate error), and the fit
  compensates by lowering h.
- ibm_fez shows 0.78% readout error on a prepared |0> (16/2048 bit-flips);
  ibm_marrakesh showed clean 2048/0. Both are consistent with real hardware;
  fez had slightly less calibrated readout at submission time.

### Honest claim

The pipeline runs end-to-end on two different IBM Quantum devices and
recovers Hamiltonian parameters to NISQ accuracy:

- J to ~1% on both backends
- h to ~13% on both backends, systematic (Trotter/gate error, not noise)

Not claimed: reproducible sub-percent precision; quantum advantage; physics
discovery. This is a portability and correctness result — the pipeline is
functional on real quantum hardware and its systematic error modes are
identifiable.

### A test bug caught by both runs

`sanity_p0` was originally thresholded at 0.25 < <Z1>|+0> < 0.75, treating
<Z1>|+0> as +0.5. That is wrong: |+> measured in Z basis gives 0. The
pipeline was correct; the human threshold was not. Both hardware runs show
<Z1>|+0> ~ 0 (from 1038/1010 and 1043/1005 splits), which is a clean
Hadamard, not a failure.
