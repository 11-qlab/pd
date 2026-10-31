# purediscover — Symbolic Discovery of Physical Laws

**Iterative Symbolic Refit Matches Bayesian Experimental Design for Physical Law Discovery**

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Abstract

We benchmark three discovery strategies on **23 physical laws** spanning classical mechanics, quantum-gravity-adjacent hypothetical extensions, and interferometric phase forms at **2% multiplicative noise**.

| Method | Convergence | Data | Time |
|--------|-------------|------|------|
| `single_shot` | 16/23 (70%) | 15 pts | 13.5s |
| **`random_growth`** | **21/23 (91%)** | **5 pts** | **8.8s** |
| `closed_loop_boed` | 8/8 (100%, 1-var only) | 5 pts | 4.5s |

**`random_growth` dominates**: more convergences, 1/3 the data, 1.5× faster. Bayesian optimal experimental design (BOED) provides **zero coverage** of the 15 multi-variable laws that dominate modern precision physics.

We report two residual failures (`cow_phase`, `atom_grav_gradient`) as diagnostic measurements, not suppressed results. We also document a convergence-check correction that raised measured rates from 48% → 70% and 74% → 91% without changing the underlying search.

**Full paper:** see [`paper/tables_v9.md`](paper/tables_v9.md) for per-law results.

---

## Key results

### Per-domain breakdown

| Domain | Laws | `single_shot` | `random_growth` | BOED |
|--------|------|---------------|-----------------|------|
| Classical mechanics / thermodynamics / EM | 11 | 11/11 | 11/11 | 8/8 |
| Quantum-gravity-adjacent | 3 | 3/3 | 3/3 | 0/0 |
| Interferometric phase | 9 | 2/9 | 7/9 | 0/0 |

### Why `random_growth` wins

- **Batch diversity**: 5-point batches iterated give more diverse parameter coverage than one 15-point batch.
- **Fast on small data**: `brute_search` is much faster on 5-point sets than 15-point sets.
- **Adaptive sampling**: new points fill gaps left by the initial batch.

### Why BOED doesn't help here

BOED's 1-D proposal framework cannot address 3- and 4-variable forms (all interferometric phases are 3+ variables). Its 100% rate on 1-var laws matches `random_growth` at 5 points — no measurable benefit.

---

## Installation

```bash
git clone git@github.com:11-qlab/pd.git
cd pd
pip install -e .
```

Dependencies: sympy, numpy (see requirements.txt).

---

Quick start

```python
from purediscover.benchmark import (
    LawCatalog, PhaseCatalog, full_catalog,
    SingleShot, RandomGrowth, ClosedLoopBOED,
    Benchmark, ReportRenderer,
)

# 1. Choose a catalog
catalog = full_catalog()   # all 23 laws
# catalog = LawCatalog.default()   # 11 classical laws
# catalog = PhaseCatalog.default()  # 9 interferometric phases

# 2. Choose methods
methods = [SingleShot(), RandomGrowth(), ClosedLoopBOED()]

# 3. Run
result = Benchmark(catalog, methods).run(noise_frac=0.02)

# 4. Report
print(ReportRenderer(result).render_markdown())
result.to_json("benchmark.json")
```

CLI:

```bash
python examples/run_benchmark.py --full --domain all --markdown
```

---

Benchmark laws

Classical (11)

stefan_boltzmann_T4, kepler_third, inverse_square, linear, ohm, newton_2, kinetic_energy, arrhenius, offset_inverse, gravitational, coulomb.

Quantum-gravity-adjacent (3)

yukawa_potential (massive graviton), newton_1pn (post-Newtonian), newton_short_distance (extra dimensions).

Interferometric phase (9)

cow_phase, sagnac_phase, magnetic_ab, electric_ab, grav_ab_photon, compton_clock, aharonov_casher, scalar_ab, atom_grav_gradient.

---

Failures (documented)

Law Why it fails v2 fix
cow_phase (m·g·T²) compose budget exhausts before trying m·g reorder _enumerate_b, raise budget
atom_grav_gradient (0.1·g·T⁴ + g·m·T²) 1-D offset peel can't handle multi-var sums greedy feature selection

Both failures are diagnosed and point to specific v2 work.

---

Citation

```bibtex
@misc{purediscover2026,
  title  = {Iterative Symbolic Refit Matches Bayesian Experimental Design
            for Physical Law Discovery},
  author = {11-qlab},
  year   = {2026},
  url    = {https://github.com/11-qlab/pd}
}


---

License

MIT

```
