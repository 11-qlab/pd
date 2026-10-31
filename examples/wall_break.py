"""Pressure test: apply brute-force BFS to each grammar wall.

Room = purediscover (no mul, no ventilator).
Pressure = BFS by node count, no early kill.
Hole = exact fit found on held-out data.
"""
import random
import time
from purediscover import Grammar
from purediscover.rich_search import brute_search


def make_data(n=40, seed=7):
    rng = random.Random(seed)
    return [
        ({"m": (m := rng.uniform(0.5, 5.0)), "a": (a := rng.uniform(0.5, 5.0))}, m * a)
        for _ in range(n)
    ]


def run(label, grammar, data, max_nodes, budget):
    t0 = time.time()
    r = brute_search(grammar, data, max_nodes=max_nodes,
                     max_per_level=20000, tol=0.02, time_budget=budget)
    dt = time.time() - t0
    if r:
        e = str(r[0]["expr"])
        nodes = r[0].get("nodes", "?")
        bits = r[0]["score"]["bits"]
        print(f"  {label:34s}  FOUND  n={nodes:>2}  bits={bits:6.1f}  {dt:5.1f}s  ->  {e[:70]}")
    else:
        print(f"  {label:34s}  (none)                                          {dt:5.1f}s")


data = make_data()

print("Pressure test (target y = m*a):")
run("add-only (true wall)",
    Grammar(["m", "a"], [1.0], ops=["add"]),
    data, max_nodes=10, budget=30)

run("add + div",
    Grammar(["m", "a"], [1.0], ops=["add", "div"]),
    data, max_nodes=6, budget=60)

run("add + sub + pow",
    Grammar(["m", "a"], [1.0], ops=["add", "sub", "pow"]),
    data, max_nodes=10, budget=600)

run("add + log + exp",
    Grammar(["m", "a"], [1.0], ops=["add", "log", "exp"]),
    data, max_nodes=6, budget=60)
