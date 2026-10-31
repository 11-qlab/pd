"""Theory generation demo on 3 known physical laws.

For each, produce a structured theory report:
    - classify form
    - compute limits
    - detect monotonicity
    - match against corpus for mechanism
    - render narrative

Optional: enable LLM mode by passing a client. Offline mode uses
only rule-based reasoning.
"""
import sympy as sp

from purediscover.theory import generate_theory, render_theory


# --- Synthetic dataset generators -----------------------------------------
def _make_data(expr, var_name, x_range, n=20, noise=0.02):
    import random
    rng = random.Random(0)
    var = sp.Symbol(var_name)
    data = []
    for _ in range(n):
        x = rng.uniform(*x_range)
        try:
            y = float(expr.subs({var: x}))
        except Exception:
            continue
        y *= (1.0 + rng.gauss(0.0, noise))
        data.append(({var_name: x}, y))
    return data


# --- Demo cases -----------------------------------------------------------
CASES = [
    {
        "name": "Arrhenius rate",
        "expr": sp.Float(1e10) * sp.exp(-sp.Float(8000) / sp.Symbol("T")),
        "var": "T",
        "range": (280.0, 400.0),
    },
    {
        "name": "Kepler",
        "expr": sp.Symbol("a") ** sp.Rational(3, 2),
        "var": "a",
        "range": (0.5, 30.0),
    },
    {
        "name": "Offset inverse",
        "expr": sp.Float(2.0) / sp.Symbol("x") + sp.Float(1.0),
        "var": "x",
        "range": (0.2, 20.0),
    },
    {
        "name": "Damped oscillation",
        "expr": (
            sp.exp(-sp.Float(0.3) * sp.Symbol("t"))
            * sp.sin(sp.Float(2.0) * sp.Symbol("t"))
        ),
        "var": "t",
        "range": (0.1, 10.0),
    },
]


def main():
    print("THEORY GENERATION DEMO")
    print("=" * 66)

    for case in CASES:
        print(f"\n>>> {case['name']}")
        data = _make_data(case["expr"], case["var"], case["range"])
        report = generate_theory(
            case["expr"], data, [case["var"]],
            x_range=case["range"],
            llm_client=None,   # rule-based mode
        )
        print(render_theory(report))

    # Show optional LLM mode with a mock
    print()
    print(">>> With LLM (mock) — showing prompt construction path")
    print("=" * 66)

    def mock_llm(prompt):
        return (
            f"[mock LLM response — prompt was {len(prompt)} chars]\n"
            "The form is consistent with thermal activation. "
            "Related to Boltzmann population and Eyring kinetics. "
            "Discriminating experiment: measure at intermediate T where "
            "Arrhenius and stretched-exponential differ most."
        )

    case = CASES[0]   # Arrhenius
    data = _make_data(case["expr"], case["var"], case["range"])
    report = generate_theory(
        case["expr"], data, [case["var"]],
        x_range=case["range"],
        llm_client=mock_llm,
    )
    print(render_theory(report))


if __name__ == "__main__":
    main()
