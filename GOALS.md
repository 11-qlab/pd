# Project Goals — Autonomous Quantum Physicist

This document states the long-term goal, the honest current state, and the
research boundaries. It is written to prevent overclaiming and to give the
project a north star while acknowledging what is achievable.

## The North Star

**Long-term goal:** a system that, given experimental quantum data with no
known closed-form law, autonomously:

1. Discovers the governing formula
2. Explains what mechanism causes it
3. Proposes the next experiment to test it
4. Reports how far / how often / how big / how fast / what if — with units
5. Distinguishes novel from rediscovered

**Nobody has this today.** Not DeepMind, not Anthropic, not OpenAI. Every
published "AI discovered X" result (AlphaFold, FunSearch, AI Feynman,
KantAM) has humans in the verification and confirmation loop. Anyone
claiming fully autonomous novel discovery is either wrong or selling.

## What is achievable — 3 tiers

### Tier 1 — Achieved (verified, benchmarked)

| Capability | Evidence | Date |
|---|---|---|
| 1D symbolic rediscovery | 15/24 Feynman quantum laws exact | committed |
| Multivariate via PySR routing | 23/55 = 41.8% symbolic on PISR quantum | committed |
| Exceeds PySR-45 SOTA baseline | +5.4 pp on same 55 laws | committed |
| Implicit F=0 via sympy.solve | Included in 23/55 | committed |
| Honest refusal when no closed form exists | Eliashberg Tc: 0/1, p90=0.36 | committed |
| Runs on real IBM Quantum hardware | ibm_marrakesh + ibm_fez, J~1% | committed |
| Aer simulation validation | Tier 1/2/3 pass | committed |

**What Tier 1 proves:** the pipeline can rediscover known quantum laws
at rates exceeding the SOTA baseline, on a published benchmark, with an
honest refusal mechanism when no closed-form law exists.

**What Tier 1 does NOT prove:** novel discovery, causal reasoning,
experiment design.

### Tier 2 — Achievable in 2–4 weeks (assisted scientist)

Wire Groq (`gpt-oss-120b`) as a hypothesis co-pilot:

| Function | Input | Output | Human verifies |
|---|---|---|---|
| `explain_law(equation, vars, domain)` | Recovered formula | Prose causal explanation | Physicist checks |
| `propose_primitive(residuals, grammar)` | Residual pattern | One new operator candidate | Pre-flight validator + MDL gate |
| `design_experiment(hypothesis, observables)` | Hypothesis + constraints | Measurement plan | Physicist runs it |
| `answer_question(question, context)` | "how far / how fast / ..." | Formula-based answer with units | Physics check |

**What Tier 2 proves:** the pipeline can *assist* a scientist — propose
and explain — while humans run the verification loop.

**What Tier 2 does NOT prove:** autonomous decision-making, actual
discovery, causal claims without verification.

### Tier 3 — Research-scope (not achievable this year)

| Capability | Why it's hard | Distance |
|---|---|---|
| Autonomous novel discovery | Requires unknown-ground-truth data + verification + literature check + independent confirmation | 5+ years for anyone |
| Autonomous experiment design in a lab | Requires robotic hardware + real-time feedback + safety | 2–3 years (chemistry/materials), longer for quantum |
| Autonomous causal reasoning | Causal inference from observational data is unsolved; LLM-generated causation is unreliable | No timeline — open problem |
| PDE symbolic regression (Schrödinger, Dirac) | Requires differential operators, boundary conditions, initial values as grammar primitives | 2–3 years for restricted cases |
| Integral-equation SR (Eliashberg) | Grammar has no `∫`; coupled self-consistent systems out of scope | 2+ years |
| Multi-equation symbolic systems | No mainstream SR does coupled equations | Active research |

## The honest boundary

**What this repo can claim today:**

> "The purediscover orchestrator routes each task class to the appropriate
> discovery strategy, recovering 23/55 = 41.8% of quantum laws on PISR's
> published benchmark by symbolic equivalence — exceeding PySR-45's
> baseline of 20/55 = 36.4% on the same laws. It runs on real IBM Quantum
> hardware (two backends, J recovered to ~1%). It refuses to fabricate a
> closed-form law when none exists (Eliashberg Tc: p90=0.36, ok=False).
> It is a discovery tool; it is not an autonomous scientist."

**What this repo cannot claim:**

- "AI discovered new quantum physics" — 23/55 are rediscoveries of known laws
- "Fully autonomous scientist" — humans close the verification loop
- "Causal reasoning" — Groq explains known physics, not discovers causes
- "16,000 quantum laws discovered" — 55 laws was the benchmark scope
- "Beats PISR directly" — PISR not run (GPU required); comparison is against PySR-45

## The realistic roadmap

**Done (Tier 1):**
- [x] 1D transform search (9 transforms)
- [x] Multivariate via PySR
- [x] Implicit F=0 via sympy
- [x] Quantum grammar with 8 custom operators
- [x] 23/55 benchmark exceeded
- [x] IBM hardware validation
- [x] Honest refusal mechanism
- [x] Eliashberg Tc correctly refused

**In progress (Tier 2):**
- [ ] `groq_copilot.py` — explain / propose / design (3 functions)
- [ ] Groq on 32 PISR failures — propose primitives, filter by MDL
- [ ] Groq explanation traces on the 23 exact laws
- [ ] Experiment design wired to `active_loop.py`

**Research-scope (Tier 3):**
- [ ] PDE symbolic regression (Schrödinger as grammar primitive)
- [ ] Integral-equation SR (Eliashberg)
- [ ] Autonomous novel discovery pipeline
- [ ] Full causal reasoning

## The honest metrics to report

| Metric | Current | Method |
|---|---|---|
| Symbolic exact rate (PISR 55) | 41.8% | `symbolic_equivalence` |
| Symbolic exact rate (Feynman 24) | 62.5% | `symbolic_equivalence` |
| Numeric p90 < 0.05 (PISR 55) | 63.6% | `p90_threshold` |
| Honest refusal rate (Eliashberg) | 100% (1/1) | `ok=False when no closed form` |
| Hardware parameter recovery (IBM) | J to ~1% | Two backends agreeing |

**Reported rates should always name the benchmark, the metric, and the scope.**
Anything less is overclaiming.

## The decision rule

Before any claim, ask:

1. **What benchmark?** (PISR 55, Feynman 24, hardware, Eliashberg)
2. **What metric?** (symbolic equivalence, p90<0.05, refusion)
3. **What scope?** (rediscovery vs novel discovery vs assisted reasoning)
4. **What's the comparison?** (baseline number on the same benchmark)

If all four cannot be named, don't make the claim.

---

*Last updated: this session. See REPORT.md for full benchmark results.
See purediscover_ext/orchestrator.py for the discovery routing logic.
See purediscover_ext/quantum_grammar.py for the quantum operator set.*
