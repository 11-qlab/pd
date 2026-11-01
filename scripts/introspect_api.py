#!/usr/bin/env python3
"""Print the real public API of the purediscover package."""
import importlib
import pkgutil
import inspect
import sys


def main() -> int:
    try:
        pkg = importlib.import_module("purediscover")
    except ImportError as e:
        print(f"FAIL: cannot import purediscover: {e}", file=sys.stderr)
        return 1

    print(f"package file : {pkg.__file__}")
    print(f"package path : {list(getattr(pkg, '__path__', []))}")
    print()
    print("=== top-level names ===")
    for name in sorted(dir(pkg)):
        if name.startswith("_"):
            continue
        obj = getattr(pkg, name)
        kind = type(obj).__name__
        print(f"  {name:32s} {kind}")
    print()

    print("=== submodules ===")
    pkg_path = list(getattr(pkg, "__path__", []))
    if pkg_path:
        for mod in pkgutil.walk_packages(pkg_path, prefix="purediscover."):
            print(f"  {mod.name}")
    print()

    print("=== callables in submodules ===")
    if pkg_path:
        for mod in pkgutil.walk_packages(pkg_path, prefix="purediscover."):
            try:
                m = importlib.import_module(mod.name)
            except Exception as e:
                print(f"  [skip] {mod.name}: {e}")
                continue
            funcs = [
                n for n in dir(m)
                if not n.startswith("_") and inspect.isfunction(getattr(m, n, None))
            ]
            if not funcs:
                continue
            print(f"  {mod.name}:")
            for f in funcs:
                try:
                    sig = str(inspect.signature(getattr(m, f)))
                except (ValueError, TypeError):
                    sig = "(?)"
                print(f"    {f}{sig}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
