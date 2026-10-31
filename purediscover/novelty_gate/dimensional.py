"""Layer 2 — dimensional consistency.

Checks that each sub-expression uses the same base dimensions in add/sub,
that exp/log/sin/cos args are dimensionless, and that nothing pathological
happens with mixed dimensions under roots.

Base dimensions (short list — extend as needed):
    M = mass, L = length, T = time, I = current, K = temperature,
    N = amount (mol), J = luminous intensity

Unknown symbols (variables not in var_dims) propagate "unknown" — the
check is silently permissive when it can't determine units.
"""
from __future__ import annotations
import sympy as sp


# Common variable → base-dimension map. Extend with your own.
DIMENSIONS: dict[str, dict[str, int]] = {
    # mechanics
    "m":   {"M": 1},
    "x":   {"L": 1},
    "d":   {"L": 1},
    "L":   {"L": 1},
    "h":   {"L": 1},
    "r":   {"L": 1},
    "R":   {"L": 1},          # radius context
    "t":   {"T": 1},
    "v":   {"L": 1, "T": -1},
    "a":   {"L": 1, "T": -2},
    "F":   {"M": 1, "L": 1, "T": -2},
    "E":   {"M": 1, "L": 2, "T": -2},
    "P":   {"M": 1, "L": 2, "T": -3},
    "p":   {"M": 1, "L": 1, "T": -1},
    "L_ang": {"M": 1, "L": 2, "T": -1},
    "k":   {"M": 1, "T": -2},  # spring constant
    "tau": {"M": 1, "L": 2, "T": -2},  # torque
    # thermodynamics
    "T":   {"K": 1},
    "Q":   {"M": 1, "L": 2, "T": -2},
    "S":   {"M": 1, "L": 2, "T": -2, "K": -1},
    # electromagnetism
    "q":   {"I": 1, "T": 1},
    "I":   {"I": 1},
    "V":   {"M": 1, "L": 2, "T": -3, "I": -1},
    "R_el":{"M": 1, "L": 2, "T": -3, "I": -2},
    "C":   {"M": -1, "L": -2, "T": 4, "I": 2},
    "B":   {"M": 1, "T": -2, "I": -1},
    "Ef":  {"M": 1, "L": 1, "T": -3, "I": -1},
    # waves/optics
    "f":   {"T": -1},
    "lam": {"L": 1},
    "w":   {"T": -1},
    # fluids
    "rho": {"M": 1, "L": -3},
    "mu":  {"M": 1, "L": -1, "T": -1},
    "A":   {"L": 2},
    "Vol": {"L": 3},
}


_UNKNOWN = None
_DIMLESS = {}


def _add_dim(a, b):
    if a is _UNKNOWN or b is _UNKNOWN:
        return _UNKNOWN
    keys = set(a) | set(b)
    out = {k: a.get(k, 0) + b.get(k, 0) for k in keys}
    return {k: v for k, v in out.items() if v != 0}


def _sub_dim(a, b):
    if a is _UNKNOWN or b is _UNKNOWN:
        return _UNKNOWN
    return _add_dim(a, {k: -v for k, v in b.items()})


def _mul_dim(a, b):
    if a is _UNKNOWN or b is _UNKNOWN:
        return _UNKNOWN
    keys = set(a) | set(b)
    out = {k: a.get(k, 0) + b.get(k, 0) for k in keys}
    return {k: v for k, v in out.items() if v != 0}


def _pow_dim(base, exp_val):
    if base is _UNKNOWN or exp_val is _UNKNOWN:
        return _UNKNOWN
    if exp_val != _DIMLESS:
        return _UNKNOWN          # rational power of dimensionful quantity
    return _UNKNOWN if base else _DIMLESS


def _dim_of(expr, var_dims) -> tuple[dict | None, str | None]:
    """Recursively compute dimensions. Returns (dims | None, error | None)."""
    if expr.is_Number:
        return _DIMLESS, None
    if expr.is_Symbol:
        name = str(expr)
        if name in var_dims:
            return var_dims[name], None
        return _UNKNOWN, None
    if expr.is_Add:  # subtraction is Add(a, Mul(-1, b)) in sympy
        dims0, err0 = _dim_of(expr.args[0], var_dims)
        if err0:
            return _UNKNOWN, err0
        for a in expr.args[1:]:
            dims_i, err_i = _dim_of(a, var_dims)
            if err_i:
                return _UNKNOWN, err_i
            if dims0 is _UNKNOWN or dims_i is _UNKNOWN:
                dims0 = _UNKNOWN
                continue
            if dims0 != dims_i:
                return _UNKNOWN, f"add/sub mismatch: {dims0} vs {dims_i}"
        return dims0, None
    if expr.is_Mul:
        out = _DIMLESS
        for a in expr.args:
            d, err = _dim_of(a, var_dims)
            if err:
                return _UNKNOWN, err
            out = _mul_dim(out, d)
        return out, None
    if expr.is_Pow:
        base, exp_ = expr.args
        d_b, err_b = _dim_of(base, var_dims)
        if err_b:
            return _UNKNOWN, err_b
        d_e, err_e = _dim_of(exp_, var_dims)
        if err_e:
            return _UNKNOWN, err_e
        return _pow_dim(d_b, d_e), None
    if expr.func in (sp.exp, sp.log, sp.sin, sp.cos, sp.tan,
                     sp.sinh, sp.cosh, sp.tanh, sp.asin, sp.acos, sp.atan):
        d, err = _dim_of(expr.args[0], var_dims)
        if err:
            return _UNKNOWN, err
        if d is _UNKNOWN:
            return _UNKNOWN, None
        if d != _DIMLESS:
            return _UNKNOWN, f"{expr.func.__name__}({d}) — arg must be dimensionless"
        return _DIMLESS, None
    if expr.func == sp.sqrt:
        d, err = _dim_of(expr.args[0], var_dims)
        if err:
            return _UNKNOWN, err
        return _pow_dim(d, _DIMLESS), None
    return _UNKNOWN, None


def dimensional_check(expr, var_dims=None, target_dims=None) -> dict:
    """Layer 2: check dimensional consistency of `expr`.

    var_dims  : dict var → dims. Missing vars are treated as unknown.
    target_dims: optional. If given, checks expr's dimensions match y's.
    """
    var_dims = var_dims or {}
    dims, err = _dim_of(expr, var_dims)
    if err:
        return {"ok": False, "reason": err, "expr_dims": None}
    if target_dims is not None and dims is not _UNKNOWN:
        if dims != target_dims:
            return {"ok": False,
                    "reason": f"expr has {dims}, target has {target_dims}",
                    "expr_dims": dims}
    return {"ok": True, "reason": "dimensionally consistent",
            "expr_dims": dims}
