#!/usr/bin/env python3
"""Convert headerless SUSY.csv (19 cols, label-first) -> clean CSV for purediscover.

Usage:
    python scripts/convert_susy.py --target M_R --n-rows 20000
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

COLS = [
    "label",
    "lep1_pt", "lep1_eta", "lep1_phi",
    "lep2_pt", "lep2_eta", "lep2_phi",
    "met", "met_phi",
    "met_rel", "axial_met", "M_R", "M_TR_2", "R", "MT2",
    "S_R", "M_Delta_R", "dPhi_r_b", "cos_theta_r1",
]

LOW = ["lep1_pt", "lep1_eta", "lep1_phi",
       "lep2_pt", "lep2_eta", "lep2_phi",
       "met", "met_phi"]

HIGH = COLS[9:]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="datasets/SUSY.csv")
    ap.add_argument("--dst", default="datasets/susy_M_R.csv")
    ap.add_argument("--target", default="M_R", choices=HIGH)
    ap.add_argument("--n-rows", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    src, dst = Path(a.src), Path(a.dst)
    if not src.exists():
        print(f"FAIL: {src} does not exist", file=sys.stderr)
        return 1

    size_gb = src.stat().st_size / 1e9
    print(f"[1/5] reading {src}  ({size_gb:.2f} GB) ...")
    df = pd.read_csv(
        src,
        header=None,
        names=COLS,
        dtype=np.float32,
        engine="c",
    )
    print(f"      rows={len(df):,}  cols={df.shape[1]}")

    print("[2/5] sanity checks ...")
    if df.shape[1] != 19:
        print(f"FAIL: expected 19 cols, got {df.shape[1]}", file=sys.stderr)
        return 1
    n_nan = int(df.isna().sum().sum())
    if n_nan:
        print(f"FAIL: {n_nan} NaN values present", file=sys.stderr)
        return 1
    print(f"      no NaNs, all 19 columns present")

    print(f"[3/5] dropping label, keeping target={a.target!r} + 8 low-level vars")
    keep = LOW + [a.target]
    out = df[keep].copy()

    if a.n_rows and a.n_rows < len(out):
        out = out.sample(n=a.n_rows, random_state=a.seed).reset_index(drop=True)
        print(f"      subsampled -> {len(out):,} rows  (seed={a.seed})")

    print(f"[4/5] writing {dst}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dst, index=False, float_format="%.10e")

    print(f"[5/5] done.")
    print()
    print(f"columns  : {', '.join(keep)}")
    print(f"target   : {a.target}")
    print(f"variables: {', '.join(LOW)}")
    print(f"file     : {dst}  ({dst.stat().st_size/1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
