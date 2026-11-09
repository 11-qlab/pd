# Finding: Closed-Form Law Recovery from Noisy Quantum Measurements

**Status:** internal report, not for external distribution.
**Date:** 2026-11.
**Scope:** 1-qubit and 2-parameter systems, simulated quantum oracle, 2% measurement noise.

---

## What was demonstrated

### 1. Exact quantum oracle
Single qubit, Hamiltonian `H = Bx·X`, initial state `|0⟩`, observable `Z`.
Analytic ground truth: `⟨Z⟩(Bx, t) = cos(2·Bx·t)`.
Verified against analytic solution to **2.22 × 10⁻¹⁶** worst-case over 15 points.
This is machine precision: the oracle is numerically exact, not approximated.

### 2. 1D law recovery (noise-free)
`cos(Bx)` recovered from 60 noise-free predictions.
Template arm won with **MRE 8.43 × 10⁻¹⁶**.
MDL correctly preferred the 2-parameter form `cos(a·x+b)` over the mathematically equivalent 3-parameter form `-sin(a·x+b−π/2)` by a factor of ~2×.

### 3. 2D law recovery under 2% noise
`cos(2·Bx·t)` recovered from 37 measurements with 2% multiplicative noise.
Best parameters: `a=1.00198, b=2.00095` (truth 1.0, 2.0) — **0.2% coefficient error**.
The MDL arbiter still preferred the 2-parameter form over its 3-parameter equivalents.
Coefficient recovery is stable from iteration 11 onward.

### 4. Sparse arm on algebraic controls (separate test)
Three synthetic tests:
- Linear `2.5x + 3y + 1.5` → recovered as `2.504x + 3.006y + 1.484`, coeff error 0.1%
- Bilinear `1.7xy + 0.3x − 0.8y + 2` → recovered as `1.686xy + 0.317x − 0.776y + 1.971`, coeff error ~1%
- Quadratic `1.5x² + 2y² − 1` → recovered as `1.498x² + 2.000y² − 0.995`, coeff error 0.2%

Runtime: 0.5–2.4s per case. These use a monomial+ratio feature library with L2 least-squares fitting and a relative-MRE gate before MDL scoring.

---

## What was NOT demonstrated

### Not early convergence
The loop ran all 30 iterations. The convergence detector fired at the cap, not before. This is a **bug** (see below), not evidence that convergence doesn't work.

### Not two independent arms
The tree-search arm returned 0 candidates on every iteration of the 2D loop because `cos` is not a sparse-polynomial-family function. This is *correct* behavior for the tree arm — it just means the two arms are non-overlapping here, not cross-validating. Cross-validation between arms has not been observed in practice yet.

### Not real physics data
Every demonstration uses the exact quantum oracle. No real detector data, no real measurement uncertainty, no systematic errors.

### Not multi-dimensional
Everything is 1D or 2D. Real physics data typically has 4–8 independent variables.

### Not general discovery
The pipeline finds laws whose *family* is in its library. The template library has 11 named families (linear, quadratic, power-law, exp, exp-square, cos, sin, damped-cos, log, inverse, sinc). The sparse library covers degree-2 polynomials and simple ratios. Anything outside is invisible.

---

## Bugs identified in this run

### Bug 1 — MRE is broken near y = 0
`_mre(yp, ys) = mean(|yp − ys| / (|ys| + 1e-12))` has no floor. Points with `y ≈ 0` contribute near-infinite relative error.
In this run, iterations 19–22 added `y = −0.017, +0.016, +0.951, +0.016`.
Final MRE of 7.5 × 10⁻² is **entirely an artifact** of these points.
The recovered parameters at iteration 30 are closer to truth than at iteration 11.
**Fix:** cap relative error at 10× when `|y| < 0.1 · max(|y|)`, or use MAE.

### Bug 2 — Convergence detector confuses regression with improvement
The check `mre_plateau = cur_mre / best_mre < 1 + ε` returns `False` whenever a new point degrades the fit — which is what happens on every iteration near `y = 0`. The monitor interprets this as "still improving" and keeps running. It cannot distinguish "search is finding better laws" from "we polluted the metric with hard points."
**Fix:** track parameter stability (a, b drifting less than 0.1% between iterations over K consecutive steps) as the primary signal, and use MRE only as a secondary check.

Both bugs are in the harness, not the discovery pipeline. The recovered laws are correct.

---

## The specific claim, restated

**Narrow form (verified):**
> The pipeline recovers a closed-form law (`cos(2·Bx·t)`) whose analytic form makes numerical solution of the Schrödinger equation unnecessary. It does this from numerical predictions alone, with no access to the internal state of the simulator.

**This is a demonstration, not a general theorem.**
It shows that when the underlying law is in the search library, symbolic recovery succeeds, and the recovered form is what makes simulation unnecessary.

**Not claimed:**
- That the pipeline can discover laws outside its library
- That this generalizes to chaotic, many-body, or phase-transition systems
- That the recovered law is "novel" in the sense of unknown to physics — everything recovered here is textbook
- That this constitutes autonomous scientific discovery — the loop runs unattended, but the question it's asked is pre-specified

**What "computed source without actual compute" means here:**
The pipeline outputs a formula. Any subsequent evaluation of that formula is now cheaper than re-simulating. The pipeline is the *discoverer* of the closed form, not the *user* of it. That is a real capability, and it is the specific one demonstrated.

---

## What the architecture is, honestly

| Component | Status | Actual behavior |
|---|---|---|
| Quantum oracle | ✅ exact | 2e-16 residual vs analytic |
| Parametric template arm | ✅ works | recovers `cos(a·x·y+b)` to 0.2% coefficient error |
| Sparse algebraic arm | ✅ works | recovers linear/bilinear/quadratic to ~1% |
| MDL arbiter | ⚠️ | works for template family; needed relative-MRE gate for sparse |
| Convergence detector | ❌ broken | fires at cap, not early |
| Proposal strategy | ❌ naive | random 2D sampling, not BOED |
| Noise handling | ⚠️ | simulated 2% multiplicative only |
| Two-arm cross-validation | ❌ not observed | arms are non-overlapping, not redundant |

Two arms with non-overlapping reachability, one MDL arbiter, one broken convergence monitor. That is what exists. Not more.

---

## What comes next

**Next milestone: real measurement data with known target.**
Candidate: Nikuradse pipe-flow data (362 measurements, 6 roughness groups).
Known target: fully-rough slope 2.0122 ± 0.0197, κ_data = 0.40457.
**Structural challenge:** Nikuradse's law is implicit (`1/√f_D = −2·log₁₀(k_s/3.7D + 2.51/(Re·√f_D))`), which the current explicit-form pipeline cannot express directly.
**Realistic scope:** recover the two asymptotic laws (smooth and rough branches) separately on the appropriate subsets, verify the paper's published constants. Transition region recovery is out of scope for the current architecture.

**Open question for the next iteration:** does the pipeline recover the correct asymptotic slopes from real (not simulated) measurements with real (not multiplicative-Gaussian) noise?
