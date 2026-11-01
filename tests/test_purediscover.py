import random
import sympy as sp

from purediscover import Grammar, discover_law, mdl_score, replay_audit
from purediscover.groups import explore_groups, _all_small_groups

def test_newton_clean():
    rng = random.Random(7)
    data = []
    for _ in range(40):
        m = rng.uniform(0.5, 5.0)
        a = rng.uniform(0.5, 5.0)
        data.append(({"m": m, "a": a}, m * a))
    g = Grammar(["m", "a"], [1.0, 2.0], max_depth=2)
    top = discover_law(g, data, n_candidates=3000, tol=0.02,
                       beam=32, time_budget=30.0)
    assert top, "no candidates"
    best = top[0]["expr"]
    for inputs, y in data:
        pred = float(best.subs({sp.Symbol(k): v for k, v in inputs.items()}))
        assert abs(pred - y) <= 0.05 * max(1.0, abs(y))

def test_mdl_constant():
    data = [({"x": i}, 5.0) for i in range(10)]
    s = mdl_score(sp.Integer(5), data, residual_tol=0.01)
    assert s["ok"]

def test_mdl_reject_noise():
    rng = random.Random(1)
    data = [({"x": x}, rng.gauss(0, 10)) for x in range(50)]
    s = mdl_score(sp.Symbol("x"), data, residual_tol=0.01)
    assert not s["ok"]

def test_groups_registry():
    groups = _all_small_groups(max_order=8)
    names = {g.name for g in groups}
    assert {"V4", "Z4xZ2", "Z2^3", "D4", "S3"} <= names

def test_groups_rank():
    def score(c):
        return (0.0 if c.abelian else 1.0) + 0.1 / c.order
    top = explore_groups(score, max_order=12, top_k=3)
    assert sum(1 for c in top if not c.abelian) >= 2
