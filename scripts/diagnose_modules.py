#!/usr/bin/env python3
import importlib, sys
from pathlib import Path

cwd = Path.cwd()
print(f"cwd: {cwd}")
print()
for name in ["purediscover", "purediscover_dual", "purediscover_quantum"]:
    d = cwd / name
    init = d / "__init__.py"
    exists_dir = d.is_dir()
    exists_init = init.is_file()
    print(f"{name:22s} dir={exists_dir}  __init__.py={exists_init}")
    if exists_dir:
        files = sorted(p.name for p in d.glob("*.py"))
        print(f"{'':22s}  files: {files}")
print()
print("sys.path[0:3]:", sys.path[:3])
