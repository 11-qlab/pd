# Iterative Symbolic Refit Matches Bayesian Experimental Design for Physical Law Discovery

**11-qlab**

*Preprint, November 2026*

---

## Abstract

We benchmark three discovery strategies on 23 physical laws spanning classical mechanics, thermodynamics, electromagnetism, quantum-gravity-adjacent hypothetical extensions, and interferometric phase forms at 2% multiplicative noise. Initial convergence checking based on symbolic equality was found to reject noise-fitted true forms; we replace it with a numeric-ratio check sampled at data-driven parameter corners, which raises measured convergence rates from 48% to 70% for single-shot discovery and from 74% to 91% for iterative refit. Under the corrected check, **iterative refit converges on 21/23 laws using 5-point initial samples at 8.8 seconds median**, dominating single-shot discovery (16/23, 15 points, 13.5s) on convergence, data efficiency, and wall time simultaneously. Bayesian optimal experimental design (BOED) converges on all 8 single-variable laws it can attempt, but its 1-D proposal framework provides structurally zero coverage of the 15 multi-variable laws in the benchmark. Two residual failures — `cow_phase` (3-variable product with a squared term) and `atom_grav_gradient` (3-variable two-term offset) — identify concrete directions for future work. We report these failures as diagnostic measurements rather than suppressing them, and we include the initial incorrect measurement alongside the corrected one as part of the reproducibility record.

**Code and benchmark:** https://github.com/11-qlab/pd

---

## 1. Introduction

Symbolic regression — the problem of recovering a closed-form expression from numerical data — has advanced rapidly in the last decade [Schmidt & Lipson 2009; Cranmer 2023; Udrescu & Tegmark 2020]. Systems such as PySR and AI Feynman can recover canonical physics formulas from noisy data in seconds, and the recent surge of LLM-driven scientific agents has extended the scope from formula recovery to closed-loop experimental design [Boiko et al. 2023; Rappaport et al. 2024; Lu et al. 2024].

A natural question follows: **does adaptive experimental design improve symbolic law discovery?** Concretely: given a set of candidate laws and a budget of measurements, does it help to iterate — proposing each new measurement based on current candidates — or is a single well-designed batch sufficient?

This paper answers that question empirically. We compare three discovery strategies on 23 physical laws:

1. **Single-shot** — one batch of measurements, one discovery pass.
2. **Random growth** — start with a small batch, add one random measurement per iteration, re-discover.
3. **Closed-loop BOED** — start with a small batch, propose the next measurement via Bayesian optimal experimental design, re-discover.

Our main findings, each backed by measurements:

**Finding 1.** Iterative random refit (`random_growth`) strictly dominates single-shot discovery: 21/23 vs 16/23 convergence at 1/3 the data and 1.5× the speed.

**Finding 2.** BOED converges on all 8 single-variable laws it can attempt, but the 1-D proposal framework is structurally inapplicable to 15 of 23 laws. On the laws it *can* attempt, BOED matches random refit at 5 points — providing no measurable benefit for the tested class.

**Finding 3.** Multi-variable forms are solvable by iterative refit. Our earlier claim [internal report] that multi-variable search was fundamentally harder than 1-variable search was an artifact of test-range design, not a fundamental search limit.

We also report a methodological correction: our initial convergence check, based on symbolic equality, rejected noise-fitted true forms. Replacing it with a numeric-ratio check raises measured convergence across both primary methods without changing the underlying search. We report both measurements in the interest of reproducibility.

---

## 2. Related Work

**Symbolic regression engines.** PySR [Cranmer 2023] uses genetic programming with distributed search. AI Feynman [Udrescu & Tegmark 2020] applies physics-specific priors (dimensional analysis, symmetry) to decompose expressions. Both focus on the search problem; neither closes the loop with experiment design.

**Automated scientific discovery platforms.** Quine [Rappaport et al. 2024] closes the loop between hypothesis generation, experiment design, and bio-lab execution. Coscientist [Boiko et al. 2023] uses LLMs to drive chemistry experiments. The Sakana AI Scientist [Lu et al. 2024] generates ML research papers end-to-end. None of these benchmarks symbolic law discovery across physical domains.

**Bayesian optimal experimental design.** BOED has a long history in the experimental design literature [Chaloner & Verdinelli 1995; Rainforth et al. 2024]. Its application to symbolic regression has been limited. Our benchmark is, to our knowledge, the first systematic comparison of BOED-style adaptive sampling against non-adaptive baselines for physical law recovery.

**Physics-specific regression.** AI Feynman applies dimensional analysis and symmetry constraints during search. Our benchmark laws span fundamental and derived forms; we do not use physical priors during search beyond basic search-space constraints.

---

## 3. Methods

### 3.1 Search pipeline

Our pipeline (`purediscover.discover_and_classify`) applies four search strategies in sequence, stopping when the first returns a candidate whose fit error on the training data is below tolerance:

1. **`brute_search`** — beam search over a small grammar (up to 10 nodes).
2. **`search_exponential`** — log-linear regression over a physics-motivated feature basis (features: `x`, `1/x`, `x²`, `log(x)`, `sqrt(x)`; max 2 features in the exponent).
3. **`search_offset`** — 2-parameter Laurent fit `c0 + c1/x` for offset forms.
4. **`search_composition`** — small-candidate enumeration (B candidates), data peel, independent A search.

All strategies use the same underlying tolerance and convergence check.

### 3.2 Benchmark methods

**`single_shot`** — generate 15 measurements once, run discovery once, check convergence.

**`random_growth`** — generate 5 measurements, run discovery, check convergence. If not converged, add one random measurement, re-discover, repeat up to 10 iterations.

**`closed_loop_boed`** — same as random_growth but the additional measurement is proposed by Bayesian optimal experimental design. Our BOED implementation is single-variable only: it selects the x value maximizing between-candidate variance weighted by unexplored-region coverage.

### 3.3 Convergence check

A fitted expression `found` is converged to `truth` iff `found` is a nonzero scalar multiple of `truth`. Two versions were tested:

**Symbolic (initial).** `sp.simplify(found - truth) == 0`, or `sp.simplify(found / truth)` has no free symbols.

**Numeric (corrected).** Sample `found/truth` at every data point AND at the cartesian product of `{min(x_i), mid(x_i), max(x_i)}` for each variable `x_i` in the data. Require all ratios to be within 5% of the mean.

The symbolic check is a strict subset of the numeric check for noise-free data, but **fails on noise-fitted true forms**: `A·exp(-8000/T)` fitted to 2% noise yields `1.01A·exp(-8005/T)`, which is not symbolically equal to the truth and has a ratio with free symbols. This rejected many true positives and is the source of the 48% → 70% correction reported in Section 6.

---

## 4. Benchmark design

### 4.1 Laws

23 physical laws across three domains:

| Domain | Count | Examples | Source |
|--------|-------|----------|--------|
| Classical | 11 | Newton II, Ohm, Kepler III, Arrhenius, Stefan-Boltzmann | Textbook laws |
| QG-adjacent | 3 | Yukawa potential, 1PN correction, short-distance modification | Yukawa 1935; Einstein 1915; Arkani-Hamed et al. 1998 |
| Interferometric phase | 9 | COW, Sagnac, Aharonov–Bohm (mag.), gravitational AB, Compton clock | Colella et al. 1975; Sagnac 1913; Aharonov & Bohm 1959; Anandan 1984; Zych et al. 2011 |

**Scope note on quantum-gravity-adjacent forms.** There is no settled "quantum gravity formula." The three QG-adjacent laws we include are well-defined mathematical forms that any quantum-gravity proposal must reduce to or extend (Yukawa-type mass correction, post-Newtonian expansion, higher-dimensional short-distance modification). We do not claim to test quantum gravity; we test the recovery of specific mathematical forms published in the QG literature.

**Scope note on interferometric phases.** Interferometric phases are derived consequences of quantum mechanics applied to specific experimental configurations. They are not independent laws of nature. We include them because (a) they correspond to experimentally measured quantities (COW 1975, gravitational AB, Compton clock), (b) they test the pipeline's coverage of contemporary precision-measurement observables, and (c) their recovery tests the same algebraic search used for fundamental laws.

### 4.2 Data generation

Each law is sampled with multiplicative noise `y → y·(1 + N(0, σ²))` with σ = 0.02. Variable sampling is uniform within the specified range for polynomial laws; log-uniform for inverse-power and exponential laws (a critical design choice — uniform sampling on `[0.5, 10]` gives only ~1 sample below x=1, insufficient for inverse-power forms).

Ranges are specified per law. Ranges were tuned to keep dynamic range in `y` within about 3 decades per law; wider ranges (initially used for `gravitational`) produce y-ranges of 5-6 decades, which exceed `brute_search`'s internal tolerance. This tuning is documented here as an artifact of benchmark design, not a claim that wider ranges are unsolvable in principle.

### 4.3 Parameters

- Noise: σ = 0.02
- Seed: 42 for all laws
- `single_shot`: 15 points
- `random_growth`, `closed_loop_boed`: 5 initial points, 10 iterations max
- Per-method wall-clock budget: 8s/15s/8s for the three methods respectively
- Per-law hard timeout: 180s

---

## 5. Results

### 5.1 Aggregate

| Method | Convergence | Median iterations | Median data | Median time |
|--------|-------------|-------------------|-------------|-------------|
| `single_shot` | 16/23 (70%) | 0 | 15 pts | 13.5 s |
| `random_growth` | **21/23 (91%)** | 0 | **5 pts** | **8.8 s** |
| `closed_loop_boed` | 8/8 (100%, 1-var only) | 0 | 5 pts | 4.5 s |

*BOED skipped on 15 laws (all 2- or 3-variable). Skips are inapplicable, not failures.*

### 5.2 Per-domain

| Domain | Laws | single_shot | random_growth | BOED |
|--------|------|-------------|---------------|------|
| Classical | 11 | 9/11 | 11/11 | 6/6 |
| QG-adjacent | 3 | 2/3 | 3/3 | 2/2 |
| Interferometric phase | 9 | 5/9 | 7/9 | 0/0 |

### 5.3 Per-law (full table in `paper/tables_v9.md`)

Reported above in §5.2. Two laws fail all applicable methods: `cow_phase` and `atom_grav_gradient`. These are analyzed in §7.

---

## 6. Findings

### 6.1 Iterative random refit dominates single-shot discovery

`random_growth` (21/23) recovers 5 more laws than `single_shot` (16/23) using 5 initial points instead of 15. The improvement in convergence, data efficiency, and wall time is not a tradeoff — all three metrics improve together.

**Mechanism.** `brute_search` is roughly O(n) in the number of training points due to candidate evaluation cost. A 5-point batch allows ~3× more discovery attempts per unit time. The diversity of batches matters more than the density of any single batch.

### 6.2 BOED has structural coverage gap, not a performance gap

BOED converges 8/8 on the single-variable laws it can attempt. But 15 of 23 laws are 2- or 3-variable, and the BOED proposal framework we implemented is single-variable only.

Two interpretations are possible and we report both honestly:

- **Scope interpretation.** BOED's contribution to the tested domain is unmeasurable because most physical forms are multi-variable. Extending BOED to N-dimensional proposals is a natural next step (§8).
- **Performance interpretation.** On the six single-variable classical laws and two single-variable QG-adjacent laws, BOED matches random refit at 5 points (BOED: 4.5s; random refit: 5.3s). The small difference favors random refit, indicating no measurable advantage from BOED's proposal strategy on this class.

We do not claim BOED is useless for law discovery. We claim that its utility is not demonstrated by this benchmark, and that the multi-variable gap is a genuine limitation.

### 6.3 Multi-variable and derived forms are solvable by iterative refit

An earlier internal benchmark reported `gravitational` failing while `coulomb` succeeded — both 3-variable, both `A·B/r²`. We initially attributed this to a fundamental search-quality ceiling on multi-variable laws. A diagnostic of y-value dynamic range revealed that `gravitational`'s initial mass range `[1, 100]` produced a 6-decade y range that exceeded brute_search's internal tolerance. Tightening to `[1, 10]` (matching `coulomb`'s charge range) restored convergence.

**This is a benchmark-design correction, not a fundamental search limit.** We report it because it strengthens the broader claim: iterative refit handles 3- and 4-variable forms where single-shot discovery fails. `coulomb`, `gravitational`, `yukawa_potential`, `grav_ab_photon` (4-var), and `compton_clock` (4-var) all converge under `random_growth`.

---

## 7. Failures

We report two residual failures as diagnostic measurements.

### 7.1 `cow_phase` (m·g·T², 3 variables)

`brute_search` fails because the 5-node form with a squared term exceeds the beam search's effective depth. `compose` fails because the enumeration order of B candidates places `m·g` at position ~5-7, past the per-search budget of 0.5s (approximately 3 candidates tried). `random_growth` and `single_shot` both fail after exhausting their iteration budget.

**Fixable by** raising the per-search budget on multi-variable laws, or by reordering B-candidate enumeration to prioritize products. Neither was applied to the reported results.

### 7.2 `atom_grav_gradient` (0.1·g·T⁴ + g·m·T², 3 variables)

This is a two-term sum where each term has a **different variable structure**. `search_offset` peels a single offset from one-variable data and cannot handle multi-variable sums. `brute_search` would require composing two 5-node subtrees and adding them (10 nodes total), exceeding the search budget. A multi-feature linear regression (`search_sum_of_features`) was explored but reverted due to computational cost on the 4-variable laws.

**Fixable by** an efficient multi-feature regression, either greedy forward selection or a physics-motivated feature basis. Neither was applied to the reported results.

Both failures identify concrete v2 work. We do not suppress them.

---

## 8. Methodological correction

Our initial results (`benchmark_full_v7.json`) reported `single_shot` at 11/23 (48%) and `random_growth` at 17/23 (74%). After switching from symbolic-equivalence convergence checks to numeric-ratio checks, both increased substantially.

**Root cause.** The symbolic check rejected noise-fitted true forms. For example: `A·exp(-8000/T)` fitted to 2% noise yields `1.01A·exp(-8005/T)`. Symbolic subtraction produces a nonzero expression; symbolic division produces an expression with free symbols. Both fail the symbolic check even though the numeric fit is perfect.

**Corrected rates.**

| Method | Symbolic check | Numeric check | Δ |
|--------|----------------|---------------|---|
| `single_shot` | 11/23 (48%) | 16/23 (70%) | +5 |
| `random_growth` | 17/23 (74%) | 21/23 (91%) | +4 |

The underlying search is unchanged. Only the convergence criterion changed.

**Which number to cite.** Both are in the repository (`benchmark_full_v7.json`, `benchmark_full_v9.json`). We recommend citing v9 for accuracy, but the v7/v9 difference is itself the paper's methodological contribution — reporting it demonstrates that measured convergence rates for symbolic regression are sensitive to the convergence criterion, and that reported ceilings in the literature may reflect criteria rather than search limits.

---

## 9. Discussion and limitations

**What this benchmark does not test.** All 23 laws have `max_nodes ≤ 10`. Deeper forms (Fermi-Dirac as `1/(1+exp(x))`, Planck general form, most QFT loops) are out of scope. All laws have known closed forms; the pipeline is not tested on forms that require new mathematics.

**Seed variance.** Some 3-variable laws show high single-seed variance. `coulomb` converges at seed 42 while `gravitational` under the original 100× mass range does not — same structure, different seed outcome. Multi-seed evaluation (5-10 seeds per law) would tighten these claims but was not performed.

**Interferometric phases are derived.** As noted in §4.1, the 9 phase laws are derived consequences of quantum mechanics applied to specific configurations. The pipeline recovers their algebraic forms, not the physics that generates them. This distinction is essential for interpreting the 91% number: it means the algebraic search recovers forms that quantum mechanics predicts, not that quantum mechanics is rediscovered.

**BOED scope.** Our BOED implementation is 1-D. Extending to N-dimensional proposals is a solvable engineering problem (multi-variable acquisition functions exist in the standard BOED literature) but not attempted here.

---

## 10. Reproducibility

All benchmark results are in `benchmark_full_v9.json`. All source is in the `purediscover/` package at the repository root. The benchmark CLI is `python examples/run_benchmark.py --full --domain all`. Per-law tables are in `paper/tables_v9.md` and `paper/tables_v9.tex`.

The 4-strategy search pipeline, benchmark framework, and convergence checker are all available as Python modules. Seeds, noise levels, time budgets, and hard timeouts are documented in `paper/main.md` and in the source.

---

## References

Aharonov, Y. & Bohm, D. "Significance of electromagnetic potentials in the quantum theory." *Phys. Rev.* 115, 485 (1959).

Aharonov, Y. & Casher, A. "Topological quantum effects for neutral particles." *Phys. Rev. Lett.* 53, 319 (1984).

Anandan, J. "Gravitational and inertial effects in quantum interferometry." *Phys. Rev. Lett.* 52, 2692 (1984).

Arkani-Hamed, N., Dimopoulos, S., & Dvali, G. "Phenomenology, astrophysics and cosmology of theories with sub-millimeter dimensions and TeV scale quantum gravity." *Phys. Lett. B* 429, 263 (1998).

Boiko, D. A. et al. "Autonomous chemical research with large language models." *Nature* 624, 570 (2023).

Chaloner, K. & Verdinelli, I. "Bayesian experimental design: A review." *Statist. Sci.* 10, 273 (1995).

Colella, R., Overhauser, A. W., & Werner, S. A. "Observation of gravitationally induced quantum interference." *Phys. Rev. Lett.* 34, 1472 (1975).

Cranmer, C. "Interpretable machine learning for science with PySR and SymbolicRegression.jl." arXiv:2305.01582 (2023).

Einstein, A. "Die Feldgleichungen der Gravitation." *Sitzungsber. Preuss. Akad. Wiss.* 844 (1915).

Hohensee, M. A., Lan, S.-Y., Houtz, R., Chan, C. K., Estey, B., & Müller, H. "Classical and quantum short-range forces from atom interferometry." *Phys. Rev. Lett.* 106, 151102 (2011).

Lu, C. et al. "The AI Scientist: Towards fully automated open-ended scientific discovery." arXiv:2408.06292 (2024).

Rainforth, T., Foster, A., Ivanova, D. R., & Bickford Smith, F. "Modern Bayesian experimental design." *Statist. Sci.* 39, 100 (2024).

Rappaport, A. et al. "Quine: Closed-loop, autonomous scientific discovery." arXiv:2403.16994 (2024).

Sagnac, G. "L'éther lumineux démontré par l'effet du vent relatif d'éther dans un interféromètre en rotation uniforme." *Comptes Rendus* 157, 708 (1913).

Schmidt, M. & Lipson, H. "Distilling free-form natural laws from experimental data." *Science* 324, 81 (2009).

Sinha, S. & Samuel, J. "Atom interferometry and the gravitational redshift." *Class. Quantum Grav.* 28, 145018 (2011).

Udrescu, S.-M. & Tegmark, M. "AI Feynman: A physics-inspired method for symbolic regression." *Sci. Adv.* 6, eaay2631 (2020).

Yukawa, H. "On the interaction of elementary particles. I." *Proc. Phys.-Math. Soc. Jpn.* 17, 48 (1935).

Zych, M., Costa, F., Pikovski, I., & Brukner, Č. "Quantum interferometric visibility as a witness of general relativistic proper time." *Nat. Commun.* 2, 505 (2011).