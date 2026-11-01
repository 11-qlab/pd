import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from purediscover_ext.physics_templates import (
    TEMPLATES, evaluate_template, check_all_templates_fail, list_templates,
    fit_template,
)
from purediscover_ext.solvers import brentq_solve


def synth(template, true_params, inputs_list):
    spec = TEMPLATES[template]
    out = []
    for inp in inputs_list:
        inputs_dict = {v: inp[i] for i, v in enumerate(spec["variables"])}
        lo, hi = spec["bracket"]
        try:
            y = brentq_solve(
                lambda yy, ins=inputs_dict: evaluate_template(template, ins, yy, true_params),
                lo, hi)
        except Exception:
            continue
        if not np.isfinite(y):
            continue
        out.append({"x": tuple(inp), "y": float(y)})
    return out


def held_out_err(template, true_p, fit_p, held_inputs):
    spec = TEMPLATES[template]
    lo, hi = spec["bracket"]
    errs = []
    for inp in held_inputs:
        inputs = {v: inp[i] for i, v in enumerate(spec["variables"])}
        try:
            y_true = brentq_solve(
                lambda yy, ins=inputs: evaluate_template(template, ins, yy, true_p),
                lo, hi)
            y_pred = brentq_solve(
                lambda yy, ins=inputs: evaluate_template(template, ins, yy, fit_p),
                lo, hi)
        except Exception:
            continue
        if np.isfinite(y_true) and np.isfinite(y_pred):
            errs.append(abs(y_pred - y_true) / max(abs(y_true), 1e-9))
    return errs


def main():
    print("Template library: %d templates" % len(TEMPLATES))
    for dom in sorted(set(s["domain"] for s in TEMPLATES.values())):
        print("  %-25s %s" % (dom, list_templates(dom)))

    tests = [
        ("antoine", [5.0, 1700.0, 230.0],
         [(300.0,), (320.0,), (340.0,), (360.0,), (380.0,)],
         [4.5, 1200.0, 180.0],
         [(310.0,), (350.0,), (370.0,)]),
        ("langmuir", [2.5, 1.5],
         [(0.1,), (0.5,), (1.0,), (2.0,), (5.0,)],
         [1.0, 1.0],
         [(0.3,), (1.5,), (3.0,)]),
        ("colebrook", [1.0, 2.0, 3.7, 2.51, 0.0],
         [(1e4, 1e-4), (3e4, 5e-4), (1e5, 1e-3), (3e5, 2e-3),
          (1e6, 5e-3), (1e4, 5e-3), (5e5, 1e-4), (5e4, 1e-4)],
         [1.0, 2.0, 3.7, 2.51, 0.0],
         [(1.5e4, 3e-4), (2e5, 8e-4), (4e5, 1.5e-3)]),
    ]

    all_ok = True
    for name, true_p, inputs, init, held_inputs in tests:
        data = synth(name, true_p, inputs)
        if not data:
            print("")
            print("%s: no synthetic data (SKIP)" % name)
            continue
        fit_p, rms = fit_template(name, data, init_params=init)
        errs = held_out_err(name, true_p, fit_p, held_inputs)
        max_err = max(errs) if errs else float("inf")
        print("")
        print("%s  (%d train, %d held)" % (name, len(data), len(errs)))
        print("  rms    = %.3e" % rms)
        print("  held-out max rel err: %.3e" % max_err)
        ok = rms < 1e-2 and max_err < 5e-2
        all_ok = all_ok and ok
        print("  fit OK: %s" % ok)

    print("")
    print("=== Novel-physics signal (langmuir data vs full library) ===")
    lang_data = synth("langmuir", [2.5, 1.5],
                      [(0.1,), (0.5,), (1.0,), (2.0,), (5.0,)])
    fails = check_all_templates_fail(lang_data, rss_floor_frac=0.05)
    print("  rejected (rss > 5pct var): %s" % fails)
    print("  langmuir correctly retained: %s" % ("langmuir" not in fails))
    print("  michaelis_menten retained (isomorphic form): %s"
          % ("michaelis_menten" not in fails))

    print("")
    print("PASS: %s" % all_ok)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
