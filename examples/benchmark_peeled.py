"""Peel-transform benchmark (imports runner from purediscover.peel)."""
from purediscover.peel import run_benchmark_peeled

if __name__ == "__main__":
    print("Peel-transform benchmark (grammar-augmented):")
    s = run_benchmark_peeled()
    print()
    print(f"Found: {s['found']}/{s['total']}  ({s['rate']*100:.1f}%)")
    print("\nBy domain:")
    for d, x in sorted(s["by_domain"].items()):
        print(f"  {d:18s} {x['found']}/{x['n']}")
