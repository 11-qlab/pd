#!/bin/bash
set -e
echo "── 1. baseline (current state) ───────────────────────────────────"
python examples/benchmark_laws.py 2>&1 | tail -15 || true

echo
echo "── 2. benchmark.py routing fix is already applied ────────────────"
echo "   (difficulty <= 3 -> discover_law ; >= 4 -> brute_search)"
echo

echo "── 3. diagnose symbolic.py ───────────────────────────────────────"
python examples/diagnose.py 2>&1 | tail -60
echo

echo "── 4. dry-run grammar patch ──────────────────────────────────────"
python patches/apply_symbolic.py
echo

read -p "apply the grammar patch? [y/N] " ans
if [[ "$ans" == "y" || "$ans" == "Y" ]]; then
    python patches/apply_symbolic.py --apply
    echo
    echo "── 5. rerun diagnose after patch ─────────────────────────────"
    python examples/diagnose.py 2>&1 | tail -40
fi

echo
echo "── 6. rerun benchmark ────────────────────────────────────────────"
python examples/benchmark_laws.py
