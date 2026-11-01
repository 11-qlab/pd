#!/usr/bin/env python3
"""Sanity-check that datasets/susy_M_R.csv is what load_csv expects."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CSV = Path("datasets/susy_M_R.csv")


def main() -> int:
    if not CSV.exists():
        print(f"FAIL: {CSV} missing — run scripts/convert_susy.py first", file=sys.stderr)
        return 1

    df = pd.read_csv(CSV)
    print(f"file  : {CSV}")
    print(f"shape : {df.shape}")
    print(f"cols  : {list(df.columns)}")
    print()

    ok = True

    if df.isna().sum().sum() != 0:
        print(f"FAIL: {int(df.isna().sum().sum())} NaNs"); ok = False
    else:
        print("PASS: no NaNs")

    if not all(np.issubdtype(t, np.number) for t in df.dtypes):
        print("FAIL: non-numeric columns present"); ok = False
    else:
        print("PASS: all columns numeric")

    if "M_R" not in df.columns:
        print("FAIL: target column 'M_R' missing"); ok = False
    else:
        print("PASS: target column present")

    print()
    print(df.describe().T.to_string())
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
