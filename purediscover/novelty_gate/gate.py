"""Novelty gate orchestrator + Layers 4 (replication) and 5 (review).

Composes Layers 1-3, plus:
    Layer 4 (replication): if user provides additional datasets, checks
        that the same form fits each independent dataset
    Layer 5 (review): renders a human-readable report with per-layer
        results and a suggestion (HIGH / MEDIUM / LOW / REJECT)
"""
from __future__ import annotations
import math
import sympy as sp

from purediscover.novelty_gate.statistical import statistical_gate
from purediscover.novelty_gate.dimensional import dimensional_check
from purediscover.novelty_gate.literature import literature_gate


# --- Layer 4: replication -----------------------------------------------

def _fit_error(expr, data) -> float:
    mags = sorted(abs(y) for _, y in data if abs(y) > 0)
    y_scale = mags[len(mags) // 2] if mags else 1.0
    floor = max(y_scale / 20.0, 1e-300)
    errs = []
    for env, y in data:
        try:
            yp = float(expr.subs({sp.Symbol(k): v for k, v in env.items()}))
        except Exception:
            return 1.0
        if not math.isfinite(yp):
            return 1.0
        errs.append(abs(y - yp) / max(abs(y), floor))
    return sum(errs) / max(1, len(errs))


def _refit_scalar(expr, var_names, data) -> sp.Expr:
    """Refit a single scalar multiplier A so A * expr.func fits data best.
    Only applies when expr has a leading scalar (up to numeric multiplier)."""
    # Determine which free symbols are variables vs fitted coefficients
    coeffs = [s for s in expr.free_symbols if str(s) not in var_names]
    if len(coeffs) > 1:
        return expr  # too many params for linear refit
    # If no coefficients, just return
    if not coeffs:
        return expr
    c = coeffs[0]
    # Compute A = sum(y * f(x)) / sum(f(x)^2) where f(x) = expr with c=1
    f_expr = expr.subs({c: 1})
    num, den = 0.0, 0.0
    for env, y in data:
        try:
            fv = float(f_expr.subs({sp.Symbol(k): v for k, v in env.items()}))
        except Exception:
            continue
        if not math.isfinite(fv):
            continue
        num += y * fv
        den += fv * fv
    if den == 0:
        return expr
    A = num / den
    return expr.subs({c: A})


def replication_gate(expr, var_names, additional_datasets, *,
                     tol: float = 0.05) -> dict:
    """Layer 4: does the same form fit each additional dataset?"""
    if not additional_datasets:
        return {"ok": None, "n_datasets": 0, "matches": [],
                "reason": "no replication data supplied — manual review needed"}
    matches = []
    for i, (data_i, vars_i) in enumerate(additional_datasets):
        if vars_i != var_names:
            matches.append({"dataset": i, "ok": False,
                            "reason": f"variable mismatch: {vars_i}"})
            continue
        refit = _refit_scalar(expr, var_names, data_i)
        err = _fit_error(refit, data_i)
        matches.append({"dataset": i, "ok": err <= tol, "fit_error": err})
    passed = sum(1 for m in matches if m["ok"])
    return {
        "ok": passed == len(matches),
        "n_datasets": len(additional_datasets),
        "n_passed": passed,
        "matches": matches,
        "reason": f"{passed}/{len(matches)} datasets fit within tol={tol}",
    }


# --- Orchestrator -------------------------------------------------------

def full_gate(expr, data, var_names, *,
              var_dims=None, target_dims=None,
              additional_datasets=None,
              n_bootstrap=30, k_folds=5, seed=0) -> dict:
    """Run all layers, return dict with each layer's result + verdict."""
    stat = statistical_gate(expr, data, var_names,
                            n_bootstrap=n_bootstrap, k_folds=k_folds, seed=seed)
    dim = dimensional_check(expr, var_dims or {}, target_dims)
    lit = literature_gate(expr)
    rep = replication_gate(expr, var_names, additional_datasets or [])

    # Verdict: weighted decision
    fails = []
    if not stat["ok"]:
        fails.append("statistical")
    if not dim["ok"]:
        fails.append("dimensional")
    if rep["ok"] is False:
        fails.append("replication")

    if not fails:
        if rep["ok"] is True and not lit["has_match"]:
            verdict = "HIGH"
        elif rep["ok"] is True or not lit["has_match"]:
            verdict = "MEDIUM"
        else:
            verdict = "LOW"
    elif len(fails) == 1 and fails[0] == "dimensional":
        verdict = "MEDIUM"   # dimensional issue is often just missing var_dims
    else:
        verdict = "REJECT"

    return {
        "expr": expr,
        "verdict": verdict,
        "fails": fails,
        "statistical": stat,
        "dimensional": dim,
        "literature": lit,
        "replication": rep,
    }


# --- Layer 5: human-readable report -------------------------------------

def _ck(b) -> str:
    if b is True: return "OK"
    if b is False: return "FAIL"
    return "—"


def render_report(gate_result: dict, show_errors: bool = False) -> str:
    r = gate_result
    expr = r["expr"]
    lines = []
    lines.append("=" * 66)
    lines.append(f"Candidate: {expr}")
    lines.append("-" * 66)

    # Statistics
    s = r["statistical"]
    cv = s["cv"]
    bs = s["bootstrap"]
    bic = s["bic"]
    lines.append("STATISTICAL")
    lines.append(f"  cross-validation : {_ck(cv['ok']):5s}  "
                 f"max CV/train = {cv.get('max_ratio', 0):.3f} "
                 f"(threshold {cv.get('threshold', 3.0)})")
    lines.append(f"  bootstrap        : {_ck(bs['ok']):5s}  "
                 f"{bs['stable_count']}/{bs['n']} resamples "
                 f"within {bs['threshold']:.4f}")
    lines.append(f"  parsimony (BIC)  : {_ck(bic['ok']):5s}  "
                 f"BIC={bic['value']:.1f}, {bic['n_params']} params")
    if show_errors and "errs" in bs:
        preview = ", ".join(f"{e:.4f}" for e in bs["errs"][:5])
        lines.append(f"    (bootstrap errors preview: {preview}...)")

    # Dimensional
    d = r["dimensional"]
    lines.append("DIMENSIONAL")
    lines.append(f"  consistency      : {_ck(d['ok']):5s}  {d['reason']}")
    if d.get("expr_dims"):
        lines.append(f"    expr_dims      = {d['expr_dims']}")

    # Literature
    l = r["literature"]
    lines.append("LITERATURE")
    n_matches = len(l["shape_matches"])
    lines.append(f"  shape fingerprint: {_ck(not l['has_match']):5s}  "
                 f"{n_matches} match(es) in extended corpus")
    for m in l["shape_matches"]:
        lines.append(f"    ✓ {m['name']:22s}  {m['description']}")

    # Replication
    rp = r["replication"]
    lines.append("REPLICATION")
    lines.append(f"  datasets         : {_ck(rp['ok']):5s}  {rp['reason']}")
    for m in rp.get("matches", []):
        lines.append(f"    dataset {m['dataset']}: "
                     f"{'OK' if m.get('ok') else 'fail'}  "
                     f"fit_error={m.get('fit_error', '?'):.4f}"
                     if m.get("fit_error") is not None
                     else f"    dataset {m['dataset']}: {m.get('reason')}")

    lines.append("-" * 66)
    verdict = r["verdict"]
    marker = {"HIGH": "★★★", "MEDIUM": "★★ ", "LOW": "★  ", "REJECT": "✗  "}
    lines.append(f"{marker[verdict]} VERDICT: {verdict}")
    if r["fails"]:
        lines.append(f"  failed layers: {', '.join(r['fails'])}")
    lines.append("=" * 66)
    return "\n".join(lines)
