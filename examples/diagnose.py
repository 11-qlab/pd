"""Diagnostic: print Grammar internals, probe ops, test known failures."""
import inspect
import time
import sympy as sp

from purediscover.symbolic import Grammar, discover_law
from purediscover.rich_search import brute_search
from purediscover.resource.laws import get
from purediscover.benchmark import _build_data, _grammar_for

print("=" * 66)
print("SIGNATURES")
print("  Grammar.__init__ :", inspect.signature(Grammar.__init__))
print("  discover_law     :", inspect.signature(discover_law))
print("  brute_search     :", inspect.signature(brute_search))
print("=" * 66)

print("GRAMMAR PROBES")
try:
    g = Grammar(variables=["x"], constants=[1.0], max_depth=2)
    print("  ops attr   :", getattr(g, "ops", "<not set>"))
    print("  _apply     :", hasattr(g, "_apply"))
    if hasattr(g, "_apply"):
        print("  _apply sig :", inspect.signature(g._apply))
        # log
        try:
            print("  _apply('log', x) ->", g._apply("log", sp.Symbol("x")), " ✓")
        except Exception as e:
            print(f"  _apply('log', x) raised: {type(e).__name__}: {e}  ✗")
        # pow exponents
        for exp in (2, 3, 4, -1, sp.Rational(1, 2)):
            try:
                r = g._apply("pow", sp.Symbol("x"), sp.Integer(exp)
                             if exp == int(exp) else sp.Rational(exp))
                print(f"  _apply('pow', x, {exp}) -> {r}")
            except Exception as e:
                print(f"  _apply('pow', x, {exp}) raised: {type(e).__name__}  ✗")
except Exception as e:
    print("  could not instantiate default Grammar:", repr(e))

print("=" * 66)
print("KINETIC_ENERGY  (target: m*v**2,  difficulty=5 -> brute_search after fix)")
law = get("kinetic_energy")
g = _grammar_for(law)
data = _build_data(law, n=40, seed=0, noise=0.0)
for name, fn in [("discover_law", lambda: discover_law(
                      g, data, n_candidates=8000, tol=0.02,
                      beam=256, time_budget=30)),
                 ("brute_search", lambda: brute_search(
                      g, data, max_nodes=12, max_per_level=50000,
                      tol=0.02, time_budget=30))]:
    print(f"  -- {name} --")
    t0 = time.time()
    try:
        cands = fn()
        print(f"     {len(cands)} candidates in {time.time()-t0:.2f}s")
        for c in cands[:3]:
            print("       ", c["expr"])
    except Exception as e:
        print("     raised:", repr(e))

print("=" * 66)
print("BOLTZMANN_ENTROPY  (target: log(W),  needs log op)")
law = get("boltzmann_entropy")
g = _grammar_for(law)
data = _build_data(law, n=40, seed=0, noise=0.0)
for name, fn in [("discover_law", lambda: discover_law(
                      g, data, n_candidates=4000, tol=0.02,
                      beam=128, time_budget=15)),
                 ("brute_search", lambda: brute_search(
                      g, data, max_nodes=6, max_per_level=10000,
                      tol=0.02, time_budget=15))]:
    print(f"  -- {name} --")
    t0 = time.time()
    try:
        cands = fn()
        print(f"     {len(cands)} candidates in {time.time()-t0:.2f}s")
        for c in cands[:3]:
            print("       ", c["expr"])
    except Exception as e:
        print("     raised:", repr(e))

print("=" * 66)
print("STEFAN_BOLTZMANN  (target: T**4,  needs pow exponent 4)")
law = get("stefan_boltzmann")
g = _grammar_for(law)
data = _build_data(law, n=40, seed=0, noise=0.0)
print("  -- brute_search --")
try:
    cands = brute_search(g, data, max_nodes=6, max_per_level=10000,
                         tol=0.02, time_budget=15)
    print(f"     {len(cands)} candidates")
    for c in cands[:3]:
        print("       ", c["expr"])
except Exception as e:
    print("     raised:", repr(e))
print("=" * 66)
