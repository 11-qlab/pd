"""Test the quantum grammar extension end-to-end.

Part 1: Synthetic Tier B tests — prove new operators unlock laws.
Part 2: PISR 55 rerun with new grammar — compare to 41.8% baseline.
"""
import json, sys, time, warnings
import numpy as np
import sympy as sp
warnings.filterwarnings("ignore")

sys.path.insert(0, ".")
from pysr import PySRRegressor
from purediscover_ext.quantum_grammar import (
    build_pysr_config, get_candidates, PHYSICS_CUSTOM, capability_report,
)

print("=== Grammar capability ===")
print(json.dumps(capability_report(), indent=2))
print()


# ============================================================
# PART 1: Synthetic Tier B tests
# ============================================================
def p90(y_pred, y_true):
    rel = np.abs(np.asarray(y_pred) - np.asarray(y_true)) / np.maximum(np.abs(y_true), 1e-10)
    return float(np.percentile(rel, 90))


def fit_pysr_synthetic(X, y, timeout_sec=60):
    cfg = build_pysr_config(timeout_sec=timeout_sec, niter=40)
    model = PySRRegressor(**cfg)
    model.fit(X, y)
    best = model.get_best()
    y_pred = model.predict(X)
    return best["equation"], p90(y_pred, y)


def tier_b_tests():
    print("=" * 70)
    print("PART 1: Synthetic Tier B tests (single-variable forms)")
    print("=" * 70)

    rng = np.random.default_rng(42)
    x = np.linspace(0.1, 5.0, 100)
    results = []

    # Q12: Bose  n = 1/(exp(x)-1)  -- x = hbar*omega/(kb*T), use x directly
    print("\nQ12 Bose-Einstein: y = 1/(exp(x)-1)")
    y = 1.0 / (np.exp(x) - 1.0)
    eq, p = fit_pysr_synthetic(x.reshape(-1, 1), y, timeout_sec=60)
    ok = p < 0.05
    print(f"  {'PASS' if ok else 'FAIL'}  p90={p:.2e}  eq: {str(eq)[:70]}")
    results.append(("Q12_bose", ok, p, str(eq)[:80]))

    # Q11: Planck radiation  u = x^3/(exp(x)-1)
    print("\nQ11 Planck radiation: y = x^3/(exp(x)-1)")
    y = x**3 / (np.exp(x) - 1.0)
    eq, p = fit_pysr_synthetic(x.reshape(-1, 1), y, timeout_sec=60)
    ok = p < 0.05
    print(f"  {'PASS' if ok else 'FAIL'}  p90={p:.2e}  eq: {str(eq)[:70]}")
    results.append(("Q11_planck", ok, p, str(eq)[:80]))

    # Q13: Fermi-Dirac  f = 1/(exp(x-mu)+1) at mu=0
    print("\nQ13 Fermi-Dirac (mu=0): y = 1/(exp(x)+1)")
    y = 1.0 / (np.exp(x) + 1.0)
    eq, p = fit_pysr_synthetic(x.reshape(-1, 1), y, timeout_sec=60)
    ok = p < 0.05
    print(f"  {'PASS' if ok else 'FAIL'}  p90={p:.2e}  eq: {str(eq)[:70]}")
    results.append(("Q13_fermi", ok, p, str(eq)[:80]))

    # Q17: Gaussian  psi = exp(-x^2/2)
    print("\nQ17 Harmonic ground: y = exp(-x^2/2)")
    x2 = np.linspace(-3, 3, 100)
    y = np.exp(-x2**2 / 2)
    eq, p = fit_pysr_synthetic(x2.reshape(-1, 1), y, timeout_sec=60)
    ok = p < 0.05
    print(f"  {'PASS' if ok else 'FAIL'}  p90={p:.2e}  eq: {str(eq)[:70]}")
    results.append(("Q17_gauss", ok, p, str(eq)[:80]))

    # Q18: Lorentz factor  gamma = 1/sqrt(1-v^2)
    print("\nQ18 Lorentz factor (c=1): y = 1/sqrt(1-v^2)")
    v = np.linspace(0.1, 0.9, 100)
    y = 1.0 / np.sqrt(1 - v**2)
    eq, p = fit_pysr_synthetic(v.reshape(-1, 1), y, timeout_sec=60)
    ok = p < 0.05
    print(f"  {'PASS' if ok else 'FAIL'}  p90={p:.2e}  eq: {str(eq)[:70]}")
    results.append(("Q18_gamma", ok, p, str(eq)[:80]))

    print(f"\nPart 1: {sum(1 for _, ok, _, _ in results if ok)}/{len(results)} synthetic Tier B laws recovered")
    return results


# ============================================================
# PART 2: PISR 55 rerun with new grammar
# ============================================================
def extract_target(truth_str, tv_keys):
    s = truth_str.replace("^", "**").strip()
    try:
        if "=" in s:
            lhs, rhs = s.split("=", 1)
            expr = sp.sympify(f"({lhs}) - ({rhs})")
        else:
            expr = sp.sympify(s)
    except Exception:
        return None
    syms = [x for x in expr.free_symbols
            if isinstance(x, sp.Symbol) and str(x) in tv_keys]
    for sym in syms:
        try:
            sols = sp.solve(sp.Eq(expr, 0), sym)
            if len(sols) == 1 and sym not in sols[0].free_symbols:
                return str(sym)
        except Exception:
            pass
    return None


def check_symbolic(eq_str, truth_str, target, inputs, X_sample, tol=1e-4):
    try:
        # Substitute x0, x1... with real symbols
        sym_map = {f"x{i}": sp.Symbol(n) for i, n in enumerate(inputs)}
        local = dict(sym_map)
        local.update({
            "square": lambda z: z**2, "cube": lambda z: z**3,
            "sqrt": sp.sqrt, "exp": sp.exp, "log": sp.log,
            "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
            "sinh": sp.sinh, "cosh": sp.cosh, "tanh": sp.tanh,
            "pi": sp.pi,
            "planck": lambda z: z / (sp.exp(z) - 1),
            "bose":   lambda z: 1 / (sp.exp(z) - 1),
            "fermi":  lambda z: 1 / (sp.exp(z) + 1),
            "gauss":  lambda z: sp.exp(-z**2),
            "sech":   lambda z: 2 / (sp.exp(z) + sp.exp(-z)),
            "gamma":  lambda z: 1 / sp.sqrt(1 - z**2),
            "coth":   lambda z: 1 / sp.tanh(z),
            "lncosh": lambda z: sp.log(sp.cosh(z)),
        })
        pred = sp.sympify(str(eq_str).replace("^", "**"), locals=local)
        t = truth_str.replace("^", "**").strip()
        if "=" in t:
            l, r = t.split("=", 1)
            truth = sp.sympify(f"({l}) - ({r})")
        else:
            truth = sp.sympify(t)
        residual = truth.subs(sp.Symbol(target), pred)
        free = list(residual.free_symbols)
        if not free:
            return abs(complex(residual.evalf())) < tol
        rng = np.random.default_rng(0)
        for _ in range(15):
            subs = {}
            for i, name in enumerate(inputs):
                subs[sp.Symbol(name)] = float(X_sample[rng.integers(len(X_sample)), i])
            try:
                v = complex(residual.subs(subs).evalf())
                pv = complex(pred.subs(subs).evalf())
                if abs(v) / max(abs(pv), 1.0) > tol:
                    return False
            except Exception:
                return False
        return True
    except Exception:
        return False


def pisr_rerun():
    print("\n" + "=" * 70)
    print("PART 2: PISR 55 quantum rerun with quantum grammar")
    print("=" * 70)

    BENCH = "data/quantum/real/pisr_src/benchmarks/merged_physics.jsonl"
    tasks = []
    with open(BENCH) as f:
        for line in f:
            r = json.loads(line)
            lid = r.get("law_id", "")
            truth = r.get("eval_truth_surface", "")
            if ("hbar" in truth or "omega_0" in truth
                    or lid.startswith(("PHONON_", "BOSE_", "SUPERCONDUCTIVITY_",
                                        "FREE_ELECTRON", "PLANCK_", "HAWKING_"))):
                tasks.append(r)

    print(f"Testing {len(tasks)} quantum laws\n")
    results = []
    t0 = time.time()

    for k, r in enumerate(tasks):
        truth = r["eval_truth_surface"]
        tv = r.get("train_values") or {}
        if not tv:
            results.append({"lid": r["law_id"], "ok": False, "skip": True})
            continue
        target = extract_target(truth, set(tv.keys()))
        if target is None or target not in tv:
            results.append({"lid": r["law_id"], "ok": False, "skip": True})
            continue
        y = np.asarray(tv[target], float)
        inputs = [s for s in tv if s != target]
        X = np.column_stack([np.asarray(tv[s], float) for s in inputs])
        n = min(800, len(y))
        X, y = X[:n], y[:n]
        mask = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
        X, y = X[mask], y[mask]
        if len(y) < 30:
            results.append({"lid": r["law_id"], "ok": False, "skip": True})
            continue

        try:
            cfg = build_pysr_config(timeout_sec=60, niter=40)
            model = PySRRegressor(**cfg)
            model.fit(X, y)
            best = model.get_best()
            eq_str = str(best["equation"])
            sym_ok = check_symbolic(eq_str, truth, target, inputs, X)
            results.append({
                "lid": r["law_id"], "ok": sym_ok,
                "p90": p90(model.predict(X), y),
                "equation": eq_str[:100],
            })
            status = "SYM✓" if sym_ok else "    "
            print(f"[{k+1:2d}/{len(tasks)}] {status}  "
                  f"{r['law_id'][:32]:32s}  p90={results[-1]['p90']:.1e}  {time.time()-t0:.0f}s")
        except Exception as e:
            results.append({"lid": r["law_id"], "ok": False, "error": str(e)[:60]})
            print(f"[{k+1:2d}/{len(tasks)}] ERR   {r['law_id'][:32]:32s}  {type(e).__name__}")

    n_pass = sum(1 for r in results if r.get("ok"))
    n_skip = sum(1 for r in results if r.get("skip"))
    print(f"\n{'='*70}")
    print(f"PISR 55 with quantum grammar: {n_pass}/{len(tasks)} symbolic")
    print(f"Prior baseline (log/sqrt/exp only): 23/55 = 41.8%")
    print(f"Delta: {n_pass - 23:+d} laws")
    print(f"{'='*70}")
    return results


if __name__ == "__main__":
    tier_b = tier_b_tests()
    pisr = pisr_rerun()

    out = {"tier_b": tier_b, "pisr_rerun": pisr}
    with open("/tmp/quantum_grammar_results.json", "w") as f:
        json.dump(out, f, indent=2)
