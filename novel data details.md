Two questions, two answers.

(Also: your batch_discover.py paste went into zsh again, not a file — no cat > ... << 'EOF' wrapper. One-liner at the bottom to save it.)

---

Part 1 — Where the code lives in your project

```
~/projects/genesis/purediscover/
│
├── purediscover/                       ← the package (importable)
│   │
│   ├── discover.py                     ★ MAIN ENTRY POINT
│   │     discover_and_classify(data, var_names)  ← the pipeline
│   │     load_csv(path, target_col)              ← CSV loader
│   │
│   ├── novelty.py                      ★ THE CLASSIFIER
│   │     classify_novelty(expr, corpus)
│   │     → returns {"class": "NOVEL"|"COMPOSITE"|"SCALED_KNOWN"|"KNOWN"}
│   │     → THIS is what decides "novel" vs "rediscovery"
│   │
│   ├── compose.py                      ← deep-target fallback (depth 3+)
│   │     search_composition(data, var_names)
│   │
│   ├── symbolic.py                     ← Grammar (variables, ops, depth)
│   ├── rich_search.py                  ← brute_search (shallow, fast)
│   │
│   ├── resource/
│   │   ├── laws.py                     ★ THE 56-LAW CORPUS
│   │   │                                 (the reference frame for novelty)
│   │   ├── lit_queries.py              ← canonical paper-title queries
│   │   ├── constants.py                ← physical constants
│   │   └── notation.py                 ← symbol synonyms
│   │
│   ├── audit.py                        ← audit gate (WITNESS/INCONCLUSIVE/CONTRADICTS)
│   ├── litcheck.py                     ← literature search (upstream)
│   ├── litcheck_named.py               ← canonical-name wrapper
│   ├── peel.py                         ← data-transform fallback
│   └── benchmark.py                    ← corpus benchmark (56/56)
│
└── examples/                            ← runnable scripts
    ├── discover_experiment.py           ← demo on 2 CSVs
    ├── make_experiment_csv.py           ← generates synthetic CSVs
    ├── novel_demo_v2.py                 ← KNOWN/COMPOSITE/NOVEL demo
    └── (many diagnostic scripts)
```

The two files that matter most for novel detection:

· purediscover/novelty.py — the classifier (what makes something NOVEL)
· purediscover/discover.py — the pipeline that calls the classifier

The file you edit to add/remove corpus laws:

· purediscover/resource/laws.py — adding a law here removes it from the NOVEL candidate pool (because now it's KNOWN).

---

Part 2 — Where to find novel CSVs (in the browser)

Novel means: the underlying functional form is NOT one of your 56 corpus laws and NOT a binary composition of two of them. The corpus covers: newton, kinetic energy, gravitation, momentum, angular momentum, centripetal, Hooke, pendulum, torque, work, distance, Coulomb, Ohm, electric power, Lorentz, Biot-Savart, capacitor/inductor energy, ideal gas, heat capacity, Carnot, Stefan-Boltzmann, Boltzmann entropy, photon energy, de Broglie, hydrogen levels, Bohr radius, emc2, energy-momentum, magnification, inverse square, wave speed, period-freq, flow rate, pressure, hydrostatic, Reynolds, bimolecular rate, pH, Michaelis-Menten, logistic growth, area/volume of basic shapes, Pythagoras, triangle area, Bayes, variance, covariance, rect perimeter, cylinder volume, dilution, compound interest.

So what's NOT in there? Laws involving exp/log/tanh of a product, laws with multiple variables in nonlinear combinations, distribution tails, damped oscillators, radiation laws, quantum corrections. That's where you should look.

Browser sources for real measured data

Source URL Best for
UCI ML Repository archive.ics.uci.edu/datasets Physics/chem measurements, curated
Kaggle kaggle.com/datasets Search "physics experiment" / "spectroscopy" / "biochemistry"
Zenodo zenodo.org Published research supplementary data
NASA Open Data data.nasa.gov Astronomy, orbital, atmospheric
CERN Open Data opendata.cern.ch Particle physics
NIST Physical Reference Data physics.nist.gov/PhysRefData High-precision constants and cross-sections
Materials Project materialsproject.org Materials science
PhysioNet physionet.org Biomedical signals
data.gov data.gov Government measurements

Specific laws to look for (not in your corpus, generate NOVEL candidates)

These have clean functional forms — good for testing whether the pipeline surfaces them as NOVEL:

Law Form Domain
Arrhenius equation k = A * exp(-Ea/(R*T)) chemistry kinetics
Beer-Lambert A = ε * c * l spectroscopy
Nernst equation E = E0 - (RT/nF) * log(Q) electrochemistry
Poiseuille's law Q = π * ΔP * r⁴ / (8 * η * L) fluid dynamics
Darcy's law Q = -k * A * ΔP / μ porous media
Fick's first law J = -D * dC/dx diffusion
Wien's displacement λ_max * T = b thermal radiation
Rayleigh-Jeans B(ν,T) = (2ν²/c²) * kT thermal radiation
Debye heat capacity C ~ T³ at low T solid state
Fermi-Dirac distribution f(E) = 1 / (1 + exp((E-μ)/kT)) quantum stat mech
Henderson-Hasselbalch pH = pKa + log([A⁻]/[HA]) acid-base
Gibbs free energy ΔG = ΔH - T*ΔS thermodynamics
Black-Scholes C = S*N(d1) - K*exp(-rT)*N(d2) finance
Larmor formula P = q²a²/(6πε₀c³) radiation
Saha ionization n₁/n₀ ∝ T^(3/2) * exp(-χ/(kT)) astrophysics
Hagen-Poiseuille ΔP = 8μLQ/(πr⁴) pipe flow
Debye-Hückel log γ = -A*z²*√I electrolyte chemistry
Log-normal distribution f(x) ∝ exp(-(log x - μ)²/(2σ²)) / x statistics

Best first targets for a NOVEL classification:

· Arrhenius — exp of -Ea/(RT), not in corpus
· Beer-Lambert — plain product ε*c*l, might classify COMPOSITE if it looks like corpus laws multiplied
· Fermi-Dirac — 1/(1+exp(...)), cannot be composed from corpus laws, definitely NOVEL
· Saha ionization — T^(3/2) * exp(...), definitely NOVEL
· Debye-Hückel — log(sqrt(I)), definitely NOVEL

Avoid unless you're testing COMPOSITE detection:

· Anything that's two corpus laws multiplied (e.g., F*d = work × force, will be COMPOSITE)
· Anything that reduces to a corpus law in a limit

How to get them into CSV format

Whichever CSV you download, you need to make it match load_csv's expected format:

```csv
col1,col2,...,y
value1,value2,...,target_value
```

Header row required. target_col is the name of the column to fit. Everything else is a variable.

Example — Arrhenius data from a kinetics experiment:

```csv
T,k
300,0.0012
310,0.0021
320,0.0038
330,0.0067
```

Then:

```python
from purediscover.discover import load_csv, discover_and_classify

data, var_names, target = load_csv("arrhenius.csv", target_col="k")
results = discover_and_classify(data, var_names,
                                time_budget_direct=30,
                                time_budget_compose=120)
```

Expected classification: NOVEL (because A*exp(-Ea/(R*T)) — with Ea/R folded into one constant — is A*exp(-k/T) in the two parameters, and exp is in the grammar but the composition exp(-k/T) needs a division inside the exp's argument, which no corpus law has).

---

Quick start — three commands

1. Save the batch script properly:

```bash
cat > examples/batch_discover.py << 'EOF'
"""Batch discovery across a directory of CSVs."""
import pathlib
from purediscover.discover import load_csv, discover_and_classify

TARGET_COL = "y"          # <-- change this to your target column name
DATA_DIR   = "datasets"   # <-- change this to your directory

for csv_path in sorted(pathlib.Path(DATA_DIR).glob("*.csv")):
    print(f"\n=== {csv_path.name} ===")
    try:
        data, vars_, target = load_csv(csv_path, target_col=TARGET_COL)
    except Exception as e:
        print(f"  skip: {e}")
        continue
    print(f"  loaded {len(data)} rows, vars={vars_}")
    results = discover_and_classify(
        data, vars_,
        time_budget_direct=30,
        time_budget_compose=90,
    )
    if not results:
        print("  (no candidates)")
        continue
    novel = [r for r in results if r["class"] == "NOVEL"]
    if novel:
        print(f"  ★ {len(novel)} NOVEL candidate(s):")
        for r in novel[:3]:
            print(f"      {r['expr']}   (fit={r['fit_error']:.4f}, "
                  f"n={r['n_nodes']})")
    else:
        top = results[0]
        print(f"  top: [{top['class']}] {top['expr']}  "
              f"(fit={top['fit_error']:.4f})")
EOF
```

2. Create the datasets directory and add one test CSV (Arrhenius — definitely NOVEL):

```bash
mkdir -p datasets
python - << 'PY'
import csv, math, random
rng = random.Random(0)
with open("datasets/arrhenius.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["T", "y"])
    A = 1e10
    Ea_over_R = 8000.0   # Kelvin
    for _ in range(40):
        T = rng.uniform(280, 400)
        k = A * math.exp(-Ea_over_R / T)
        k *= (1 + rng.gauss(0, 0.01))
        w.writerow([f"{T:.2f}", f"{k:.6e}"])
print("wrote datasets/arrhenius.csv  (target: y = A*exp(-Ea/(R*T)))")
PY
```

3. Run:

```bash
python examples/batch_discover.py
```

Expected output:

```
=== arrhenius.csv ===
  loaded 40 rows, vars=['T']
  ★ 1 NOVEL candidate(s):
      A*exp(-Ea_over_R/T)   (fit=0.0075, n=7)
```

If that shows up, the pipeline is finding real functional forms outside the corpus.

---

The one thing to be clear about before you run this

NOVEL ≠ new physics. NOVEL means "not in your 56-law corpus." Arrhenius is one of the most well-known laws in all of chemistry — the pipeline will find it as NOVEL because Arrhenius isn't in your 56. That's the correct semantic. The pipeline's job is to find candidate functional forms from data; identifying which ones are genuinely novel to science requires the litcheck audit + human review.

For your first real experiment:

1. Grab one real dataset from UCI/Kaggle/Zenodo with a clear functional form
2. Run batch_discover.py with TARGET_COL set correctly
3. Look for NOVEL in the output
4. If a NOVEL form appears that you recognize → pipeline works, and that's your calibration case
5. If it appears repeatedly across multiple datasets of the same phenomenon → that's your genuine discovery candidate

The whole loop from "raw data" → "candidate law" is now one function call. The rest is data selection.







What "novel" means now, honestly

NOVEL means "not in your 56-law corpus and not a binary composition of two corpus laws." Arrhenius and Kepler are both human-known laws — but they're genuinely novel to your corpus, and the pipeline flagged them correctly. That's the correct semantic.
Genuine scientific novelty — a law nobody has published — is one step further: it requires audit_literature(expr) returning INCONCLUSIVE from healthy sources, replication across independent datasets, and human review. The pipeline doesn't claim to reach that; it makes the candidate cheap to find and hand off.

