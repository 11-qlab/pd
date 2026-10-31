"""Isolated check: which peel recovers sqrt(L), and why did square fail before?"""
from purediscover.resource.laws import get
from purediscover.benchmark import _build_data, _grammar_for, _equivalent_up_to_scalar
from purediscover.peel import PEELS, _augment_for_peel

law = get("pendulum_period")
data = _build_data(law, n=40, seed=0)
g = _grammar_for(law)
g2 = _augment_for_peel(g)
print(f"target:        {law.expr}")
print(f"grammar ops:   {g.ops}")
print(f"augmented ops: {g2.ops}")
print()
for name, fn in PEELS:
    cands, _ = fn(g, data, 20.0, 12, 0.02)
    status = ""
    if cands:
        best = cands[0]["expr"]
        ok = _equivalent_up_to_scalar(best, law.expr)
        status = f"best={best}  match={ok}"
    print(f"  {name:10s}: {len(cands)} candidates  {status}")
