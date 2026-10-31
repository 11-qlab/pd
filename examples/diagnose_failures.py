"""Reproduce the two concrete failures from the benchmark."""
import traceback
import sympy as sp

from purediscover.benchmark.catalog import get_law, make_data
from purediscover.discover import discover_and_classify

print("=" * 66)
print("BUG 1: arrhenius IndexError")
print("=" * 66)

law = get_law("arrhenius")
data = make_data(law, 15, 0.02)
print(f"  N = {len(data)}")
ys = [y for _, y in data]
print(f"  y range = {min(ys):.3e} .. {max(ys):.3e}")

try:
    r = discover_and_classify(
        data, law.var_names,
        time_budget_direct=4, time_budget_compose=4,
    )
    print(f"  no crash — {len(r)} candidates")
except Exception:
    print("  CRASH:")
    traceback.print_exc()

print()
print("=" * 66)
print("BUG 2: gravitational KeyError 'm2' in random_growth")
print("=" * 66)

law2 = get_law("gravitational")
data2 = make_data(law2, 5, 0.02)
print(f"  var_names = {law2.var_names}")
print(f"  N = {len(data2)}")
print(f"  first env keys = {list(data2[0][0].keys())}")

from purediscover.benchmark.runner import run_random_growth
try:
    result = run_random_growth(law2, initial_n=5, max_iters=2, budget=4.0)
    print(f"  no crash — result = {result}")
except Exception:
    print("  CRASH:")
    traceback.print_exc()
