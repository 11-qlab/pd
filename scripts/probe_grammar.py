#!/usr/bin/env python3
"""Inspect purediscover.Grammar — what operators does it actually support?"""
import inspect
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from purediscover import Grammar


def main() -> int:
    print("=== Grammar class ===")
    print(f"signature: {inspect.signature(Grammar)}")
    print()

    try:
        src = inspect.getsource(Grammar)
        print("--- source (truncated) ---")
        print(src[:2500])
        print("--- end source ---")
    except Exception as e:
        print(f"(source unavailable: {e})")
    print()

    print("=== instance attributes ===")
    try:
        g = Grammar()
    except TypeError as e:
        print(f"Grammar() needs args: {e}")
        return 1
    attrs = [a for a in dir(g) if not a.startswith("_")]
    print(f"attrs: {attrs}")
    for name in ("ops", "operators", "primitives", "functions", "fns"):
        if hasattr(g, name):
            print(f"  .{name} = {getattr(g, name)!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
