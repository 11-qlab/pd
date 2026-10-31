"""Layer 5 — Theory generation.

Two modes:

    Rule-based (offline, no API):
        - classify expression form
        - compute limits (x→0, x→∞)
        - detect monotonicity
        - match against corpus for mechanism
        - produce a structured theory report

    LLM-assisted (optional):
        - same analysis, then prompt an LLM with structured context
        - LLM proposes physical mechanism
        - caller supplies llm_client(prompt) -> str

The rule-based mode is the default. It works without any API key.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import sympy as sp


@dataclass
class TheoryReport:
    expression: str
    shape_class: str
    limits: dict
    monotonicity: str
    corpus_match: dict | None
    narrative: str
    llm_hypothesis: str | None = None
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Symbolic analysis
# ---------------------------------------------------------------------------

def _try_limit(expr, var, to):
    try:
        val = sp.limit(expr, var, to)
        if val.is_finite or val in (sp.oo, -sp.oo, sp.zoo):
            return str(val)
    except Exception:
        pass
    return None


def _detect_monotonicity(expr, var, x_range=(0.2, 20.0)) -> str:
    try:
        d = sp.diff(expr, var)
    except Exception:
        return "unknown"
    signs = []
    for xv in np.linspace(x_range[0], x_range[1], 15):
        try:
            dv = float(d.subs(var, xv))
        except Exception:
            continue
        if not math.isfinite(dv):
            continue
        signs.append(math.copysign(1, dv))
    if not signs:
        return "unknown"
    pos = sum(1 for s in signs if s > 0)
    neg = sum(1 for s in signs if s < 0)
    if pos == len(signs):
        return "monotonically increasing"
    if neg == len(signs):
        return "monotonically decreasing"
    if pos > 0 and neg > 0:
        return "non-monotone"
    return "constant"


def _classify_shape(expr, var) -> str:
    """Return a short natural-language description of the form class."""
    syms = expr.free_symbols - {var}
    has_exp = any(e.func == sp.exp for e in expr.atoms(sp.exp))
    has_log = any(e.func == sp.log for e in expr.atoms(sp.log))
    has_trig = any(
        e.func in (sp.sin, sp.cos, sp.tan)
        for e in expr.atoms(sp.sin, sp.cos, sp.tan)
    )
    has_inverse = any(
        e.is_Pow and len(e.args) == 2 and e.args[0] == var
        and e.args[1].is_negative
        for e in expr.atoms(sp.Pow)
    )

    # Power law detection — must come BEFORE has_exp/has_inverse checks.
    # Catches a**(3/2), 1.5*x**2, x, a**(29/19).
    try:
        coeff, rest = expr.as_coeff_Mul()
        if rest.is_Pow and len(rest.args) == 2 and rest.args[0] == var:
            exp = rest.args[1]
            if exp.is_rational:
                return f"power law (exponent {exp})"
        if rest == var:
            return "linear"
    except Exception:
        pass

    # Polynomial (integer powers of var, coefficients are numbers)
    try:
        poly = sp.Poly(expr, var)
        if poly.degree() > 0 and all(
            not term.free_symbols - {var} for term in poly.terms()
        ):
            return f"polynomial (degree {poly.degree()})"
    except Exception:
        pass

    if has_trig:
        return "oscillatory"
    if has_exp and has_inverse:
        return "exponential-inverse hybrid"
    if has_exp:
        return "exponential"
    if has_log:
        return "logarithmic"
    if has_inverse:
        return "inverse power law (1/x form)"
    if syms:
        return "composite with free coefficients"
    return "unclassified"


# ---------------------------------------------------------------------------
# Corpus match
# ---------------------------------------------------------------------------

def _match_corpus(expr) -> dict | None:
    """Check algebraic fingerprint against extended corpus."""
    try:
        from purediscover.novelty_gate.literature import (
            algebraic_fingerprint, _extended_laws,
        )
    except Exception:
        return None
    try:
        fp = algebraic_fingerprint(expr)
    except Exception:
        return None
    for name, known_expr, var_names, desc in _extended_laws():
        try:
            if algebraic_fingerprint(known_expr) == fp:
                return {
                    "name": name,
                    "description": desc,
                    "variables": var_names,
                }
        except Exception:
            continue
    return None


# ---------------------------------------------------------------------------
# Narrative builder
# ---------------------------------------------------------------------------

def _build_narrative(expr, shape, limits, monot, corpus) -> str:
    lines = []
    lines.append(f"Expression: {expr}")
    lines.append(f"Form class: {shape}")
    lines.append(f"Behavior: {monot} over the sampled range")
    if limits.get("x→0") or limits.get("x→∞"):
        parts = []
        if limits.get("x→0"):
            parts.append(f"as x→0, y→{limits['x→0']}")
        if limits.get("x→∞"):
            parts.append(f"as x→∞, y→{limits['x→∞']}")
        lines.append(f"Limits: {'; '.join(parts)}")
    if corpus:
        lines.append(
            f"Corpus match: {corpus['name']} — {corpus['description']}"
        )
    else:
        lines.append("Corpus match: none (potentially novel form)")

    # Interpretation hints
    if shape.startswith("exponential"):
        if "increasing" in monot:
            lines.append(
                "Interpretation: growth process; consistent with thermal "
                "activation, autocatalysis, or positive feedback."
            )
        elif "decreasing" in monot:
            lines.append(
                "Interpretation: decay process; consistent with radioactive "
                "decay, cooling, or damping."
            )
    elif shape.startswith("inverse power"):
        lines.append(
            "Interpretation: 1/r^n dependence; consistent with a point "
            "source, Coulomb-like field, or geometric spreading."
        )
    elif shape.startswith("polynomial"):
        lines.append(
            "Interpretation: algebraic relation; check whether higher-order "
            "terms are physically meaningful or curve-fit artifacts."
        )
    elif shape == "oscillatory":
        lines.append(
            "Interpretation: periodic behavior; consistent with waves, "
            "oscillators, or rotational symmetry."
        )
    elif shape == "logarithmic":
        lines.append(
            "Interpretation: logarithmic response; often appears in "
            "information, entropy, and saturation contexts."
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM hook
# ---------------------------------------------------------------------------

def _build_llm_prompt(expr, shape, limits, monot, corpus, data) -> str:
    lines = [
        "You are analyzing a candidate physical law discovered from data.",
        "",
        f"Expression: {expr}",
        f"Form class: {shape}",
        f"Monotonicity: {monot}",
        f"Limits: x→0: {limits.get('x→0')}  |  x→∞: {limits.get('x→∞')}",
    ]
    if corpus:
        lines.append(f"Matches known corpus entry: {corpus['name']} "
                     f"({corpus['description']})")
    else:
        lines.append("No corpus match — potentially novel form.")
    lines.append("")
    lines.append(f"Sample data ({len(data)} points, first 5):")
    for env, y in data[:5]:
        parts = ", ".join(f"{k}={v:.3g}" for k, v in env.items())
        lines.append(f"  ({parts}) → {y:.4g}")
    lines.append("")
    lines.append("Provide:")
    lines.append("1. The most plausible physical mechanism producing this form.")
    lines.append("2. Two related phenomena in the literature.")
    lines.append("3. One experiment that would discriminate this law from a close alternative.")
    lines.append("")
    lines.append("Be concise. Do not claim this is novel — only explain what it could be.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_theory(
    expr,
    data,
    var_names,
    *,
    x_range: tuple[float, float] | None = None,
    llm_client=None,
) -> TheoryReport:
    """Produce a structured theory report for a discovered expression.

    Args:
        expr       : sympy expression (fitted, numeric coefficients)
        data       : list of (env_dict, y) tuples
        var_names  : list of variable names (single-variable supported)
        x_range    : (min, max) for the variable; inferred from data if None
        llm_client : optional callable(prompt) -> str; if None, rule-based only

    Returns:
        TheoryReport
    """
    if len(var_names) != 1:
        raise NotImplementedError("Theory generation currently supports single-variable data")

    var = sp.Symbol(var_names[0])
    if x_range is None:
        xs = [env[var_names[0]] for env, _ in data]
        x_range = (min(xs), max(xs))

    # Analysis
    limits = {
        "x→0": _try_limit(expr, var, 0),
        "x→∞": _try_limit(expr, var, sp.oo),
    }
    monot = _detect_monotonicity(expr, var, x_range)
    shape = _classify_shape(expr, var)
    corpus = _match_corpus(expr)

    # Narrative
    narrative = _build_narrative(expr, shape, limits, monot, corpus)

    # Optional LLM
    llm_hyp = None
    if llm_client is not None:
        prompt = _build_llm_prompt(expr, shape, limits, monot, corpus, data)
        try:
            llm_hyp = llm_client(prompt)
        except Exception as e:
            llm_hyp = f"(LLM call failed: {type(e).__name__}: {e})"

    return TheoryReport(
        expression=str(expr),
        shape_class=shape,
        limits=limits,
        monotonicity=monot,
        corpus_match=corpus,
        narrative=narrative,
        llm_hypothesis=llm_hyp,
    )


def render_theory(report: TheoryReport) -> str:
    lines = []
    lines.append("=" * 66)
    lines.append("THEORY REPORT")
    lines.append("-" * 66)
    lines.append(report.narrative)
    if report.llm_hypothesis:
        lines.append("")
        lines.append("LLM-ASSISTED HYPOTHESIS")
        lines.append("-" * 66)
        lines.append(report.llm_hypothesis)
    lines.append("=" * 66)
    return "\n".join(lines)


__all__ = ["generate_theory", "render_theory", "TheoryReport"]
