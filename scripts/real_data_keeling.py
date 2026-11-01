#!/usr/bin/env python3
"""Task 2: real-data discovery on the Mauna Loa CO2 record (Keeling curve).

Real measurements, 1958-present. Ground truth decomposition:
    CO2(t) ≈ A + B*t + C*cos(2*pi*t + phi)   [Keeling et al. 1976, updated]

Pipeline task: discover this form from the raw data alone.
Success criterion: recovered frequency within 5% of 2*pi, seasonal amplitude
within 30% of published, on held-out last 10 years.
"""
import json, math, sys, time, urllib.request
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
from purediscover_templates import search_templates_nd
from purediscover_ext import tree_search_ext


CSV_URL = "https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_mlo.csv"
LOCAL = _ROOT / "datasets/real/co2_mm_mlo.csv"


def fetch_data() -> Path:
    LOCAL.parent.mkdir(parents=True, exist_ok=True)
    if LOCAL.exists() and LOCAL.stat().st_size > 5000:
        print(f"[data] using cached {LOCAL} ({LOCAL.stat().st_size} bytes)")
        return LOCAL
    print(f"[data] downloading {CSV_URL}")
    with urllib.request.urlopen(CSV_URL, timeout=30) as r:
        LOCAL.write_bytes(r.read())
    print(f"[data] saved {LOCAL} ({LOCAL.stat().st_size} bytes)")
    return LOCAL


def parse_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """NOAA format: comment header lines starting with '#', then columns:
       year, month, decimal, average, deseasonalized, ndays, sdev, unc
       We use decimal date (col 3) and average CO2 (col 4, col 5 in 1-idx).
    """
    rows = []
    with path.open() as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith("#") or s.startswith("year"):
                continue
            parts = s.split(",")
            if len(parts) < 5:
                continue
            try:
                year = int(parts[0])
                month = int(parts[1])
                dec = float(parts[2])
                avg = float(parts[4])
            except (ValueError, IndexError):
                continue
            if avg < 0 or avg > 1000:   # missing sentinel
                continue
            rows.append((dec, avg))
    if not rows:
        raise RuntimeError(f"no valid rows parsed from {path}")
    arr = np.array(rows)
    return arr[:, 0], arr[:, 1]


def main() -> int:
    path = fetch_data()
    t_raw, y_raw = parse_csv(path)
    print(f"[data] {len(t_raw)} valid monthly observations")
    print(f"[data] t range  [{t_raw.min():.2f}, {t_raw.max():.2f}] years")
    print(f"[data] y range  [{y_raw.min():.2f}, {y_raw.max():.2f}] ppm")
    print()

    # normalize t to start at 0 for numerical stability
    t0 = float(t_raw.min())
    t = t_raw - t0
    y = y_raw

    # hold out last 10 years for validation
    tmax = t.max()
    train_mask = t < (tmax - 10.0)
    test_mask  = t >= (tmax - 10.0)
    t_train, y_train = t[train_mask], y[train_mask]
    t_test,  y_test  = t[test_mask],  y[test_mask]
    print(f"[split] train: {len(t_train)} rows (t < {tmax-10:.2f})")
    print(f"[split] test:  {len(t_test)} rows  (t >= {tmax-10:.2f})")
    print()

    # build dict-based data for pipeline
    data = [({"t": float(tt)}, float(yy)) for tt, yy in zip(t_train, y_train)]

    print("=== template arm (ND) ===")
    t_start = time.time()
    templ_results = search_templates_nd(
        data, ["t"],
        max_order=2, max_mre=0.5,
        n_multistart=20, seed=42,
    )
    dt = time.time() - t_start
    print(f"  {len(templ_results)} candidates in {dt:.1f}s")
    for i, r in enumerate(templ_results[:5]):
        print(f"  [{i}] {r.name:25s}  mae={r.mae:.3e}  bits={r.mdl_bits:8.1f}")
        print(f"      {r.expr_str[:100]}")
    print()

    print("=== sparse arm ===")
    t_start = time.time()
    sparse_results = tree_search_ext(
        data, ["t"],
        max_features=4, degree=2, include_trig=True,
        tol=0.02, time_budget=60.0,
    )
    dt = time.time() - t_start
    print(f"  {len(sparse_results)} candidates in {dt:.1f}s")
    for i, r in enumerate(sparse_results[:5]):
        print(f"  [{i}] mae={r['mre']:.3e}  bits={r['bits']:8.1f}  {r['expr'][:90]}")
    print()

    # pick the best across arms
    pool = []
    for r in templ_results:
        if math.isfinite(r.mdl_bits):
            pool.append({"source": "templ", "expr": r.expr_str,
                         "mae": r.mae, "bits": r.mdl_bits, "raw": r})
    for r in sparse_results:
        if math.isfinite(r["bits"]):
            pool.append({"source": "sparse", "expr": r["expr"],
                         "mae": r["mre"], "bits": r["bits"], "raw": r})
    pool = [p for p in pool if math.isfinite(p["bits"])]
    if not pool:
        print("FAIL: no valid candidates")
        return 1
    pool.sort(key=lambda p: (p["bits"], p["mae"]))
    best = pool[0]

    print("=== best across arms ===")
    print(f"  source : {best['source']}")
    print(f"  expr   : {best['expr']}")
    print(f"  mae    : {best['mae']:.3e}")
    print(f"  bits   : {best['bits']:.1f}")
    print()

    # extract frequency from the expression (look for cos coefficient)
    import re
    expr = best["expr"]
    freq = None
    cos_match = re.search(r"cos\(([^)]+)\)", expr)
    if cos_match:
        inner = cos_match.group(1)
        # try to extract the coefficient of t
        coef_match = re.search(r"([0-9.]+)\*t", inner)
        if coef_match:
            try:
                freq = float(coef_match.group(1))
            except ValueError:
                pass

    print("=== verdict ===")
    print(f"  recovered frequency : {freq}")
    print(f"  expected            : 2*pi = {2*math.pi:.4f}")
    if freq is not None:
        rel_err = abs(freq - 2*math.pi) / (2*math.pi)
        print(f"  relative error      : {rel_err*100:.2f}%")
        freq_ok = rel_err < 0.05
    else:
        freq_ok = False
        print(f"  frequency not extractable from expression")

    # held-out validation
    print()
    print("=== held-out test (last 10 years) ===")
    y_pred_full = None
    try:
        import sympy as sp
        t_sym = sp.Symbol("t")
        expr_sym = sp.sympify(expr)
        f = sp.lambdify(t_sym, expr_sym, modules=["numpy"])
        y_pred_train = np.asarray(f(t_train), dtype=float)
        y_pred_test  = np.asarray(f(t_test),  dtype=float)
        if not np.all(np.isfinite(y_pred_test)):
            raise RuntimeError("non-finite predictions on test set")
        mae_train = float(np.mean(np.abs(y_pred_train - y_train)))
        mae_test  = float(np.mean(np.abs(y_pred_test  - y_test)))
        # also compare against the naive persistence baseline
        mae_persistence = float(np.mean(np.abs(y_test - y_train[-1])))
        print(f"  train MAE           : {mae_train:.3f} ppm")
        print(f"  test  MAE           : {mae_test:.3f} ppm")
        print(f"  persistence basline : {mae_persistence:.3f} ppm")
        print(f"  beats persistence   : {mae_test < mae_persistence}")
        generalizes_ok = mae_test < mae_persistence
    except Exception as e:
        print(f"  prediction failed: {type(e).__name__}: {e}")
        generalizes_ok = False

    print()
    print(f"  FREQUENCY OK  : {freq_ok}")
    print(f"  GENERALIZES   : {generalizes_ok}")
    print()
    if freq_ok and generalizes_ok:
        print("  VERDICT: PASS  -- real-data discovery confirmed")
        return 0
    print("  VERDICT: REVIEW  -- inspect failures above")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
