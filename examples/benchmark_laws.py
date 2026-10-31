"""Rediscovery benchmark against the known-law corpus."""
from purediscover.benchmark import run_all

print("Rediscovery benchmark (target = law's stripped form):")
summary = run_all(time_budget=30.0, max_difficulty=7)
print()
print(f"Found: {summary['found']}/{summary['total']}  ({summary['rate']*100:.1f}%)")
print("\nBy domain:")
for d, s in sorted(summary["by_domain"].items()):
    print(f"  {d:18s} {s['found']}/{s['n']}")
