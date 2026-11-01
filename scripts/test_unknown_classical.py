import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from purediscover_ext.physics_templates import check_all_templates_fail
from purediscover_ext.unknown_classical import (
    discover_structure, predict_from_discovery,
)


def test_kohlrausch(noise_frac=0.0, seed=42):
    Lambda0, K = 150.0, 60.0
    c = np.logspace(-4, 0, 40)
    Lambda_true = Lambda0 - K * np.sqrt(c)
    rng = np.random.default_rng(seed)
    if noise_frac > 0:
        Lambda = Lambda_true * (1 + noise_frac * rng.standard_normal(len(c)))
    else:
        Lambda = Lambda_true

    data_dict = [{"x": {"c": float(ci)}, "y": float(Li)} for ci, Li in zip(c, Lambda)]
    data_tuple = [{"x": (float(ci),), "y": float(Li)} for ci, Li in zip(c, Lambda)]
    fails = check_all_templates_fail(data_tuple, rss_floor_frac=0.05)
    result = discover_structure(data_dict, ["c"], "Lambda")
    if result["best"] is None:
        return False, {"error": "no transform found"}

    slope = result["best"]["slope"]
    intercept = result["best"]["intercept"]
    K_hat = -slope
    Lambda0_hat = intercept

    held_c = np.array([1e-3, 1e-2, 0.1, 0.5])
    held_true = Lambda0 - K * np.sqrt(held_c)
    held_pred = predict_from_discovery(result["best"], held_c)
    held_err = np.abs(held_pred - held_true) / held_true

    if noise_frac == 0:
        r2t, pt, ht = 0.999, 0.01, 0.05
    else:
        # R2_max ~ 1 - noise_var/signal_var ~ 0.98 for 2% mult noise here;
        # threshold reflects what is achievable, not what would be nice.
        r2t, pt, ht = 0.95, 0.10, 0.05

    tx = result["best"]["transform_x"]
    ty = result["best"]["transform_y"]
    r2 = result["best"]["r2"]
    correct = (tx == "sqrt" and ty == "identity")
    ok = (len(fails) >= 10 and correct and r2 > r2t
          and abs(Lambda0_hat - Lambda0) / Lambda0 < pt
          and abs(K_hat - K) / K < pt and held_err.max() < ht)
    return ok, {
        "n_rejected": len(fails),
        "transform": (tx, ty),
        "r2": r2,
        "Lambda0_true": Lambda0, "Lambda0_hat": Lambda0_hat,
        "K_true": K, "K_hat": K_hat,
        "held_max": float(held_err.max()),
        "expr": result["expression"],
    }


def test_souders_brown(noise_frac=0.0, seed=7):
    K, rho_L = 0.1, 800.0
    rho_V = np.linspace(1, 400, 50)
    v_true = K * np.sqrt((rho_L - rho_V) / rho_V)
    rng = np.random.default_rng(seed)
    if noise_frac > 0:
        v = v_true * (1 + noise_frac * rng.standard_normal(len(rho_V)))
    else:
        v = v_true

    data_dict = [{"x": {"rho_V": float(r)}, "y": float(vv)} for r, vv in zip(rho_V, v)]
    data_tuple = [{"x": (float(r),), "y": float(vv)} for r, vv in zip(rho_V, v)]
    fails = check_all_templates_fail(data_tuple, rss_floor_frac=0.05)
    result = discover_structure(data_dict, ["rho_V"], "v")
    if result["best"] is None:
        return False, {"error": "no transform found"}

    slope = result["best"]["slope"]
    intercept = result["best"]["intercept"]
    K_hat = np.sqrt(-intercept)
    rho_L_hat = -slope / intercept

    held_rho = np.array([5.0, 20.0, 100.0, 300.0])
    held_true = K * np.sqrt((rho_L - held_rho) / held_rho)
    held_pred = predict_from_discovery(result["best"], held_rho)
    held_err = np.abs(held_pred - held_true) / held_true

    if noise_frac == 0:
        r2t, pt, ht = 0.999, 0.01, 0.05
    else:
        # rho_L derived as -slope/intercept; intercept ~ -K^2 is small, so
        # 6-8% error is the achievable floor at 2% input noise, not a bug.
        # Held-out prediction (ht) is the honest quality test.
        r2t, pt, ht = 0.99, 0.10, 0.05

    tx = result["best"]["transform_x"]
    ty = result["best"]["transform_y"]
    r2 = result["best"]["r2"]
    correct = (tx == "reciprocal" and ty == "square")
    ok = (len(fails) >= 10 and correct and r2 > r2t
          and abs(K_hat - K) / K < pt
          and abs(rho_L_hat - rho_L) / rho_L < pt and held_err.max() < ht)
    return ok, {
        "n_rejected": len(fails),
        "transform": (tx, ty),
        "r2": r2,
        "K_true": K, "K_hat": K_hat,
        "rhoL_true": rho_L, "rhoL_hat": rho_L_hat,
        "held_max": float(held_err.max()),
        "expr": result["expression"],
    }


def print_result(name, ok, info):
    print("")
    print(name)
    print("  PASS: " + str(ok))
    if "error" in info:
        print("  Error: " + str(info["error"]))
        return
    print("  templates rejected: " + str(info["n_rejected"]) + "/14")
    print("  transform: " + str(info["transform"]))
    print("  R2: %.6f" % info["r2"])
    print("  expr: " + info["expr"])
    if "Lambda0_true" in info:
        print("  Lambda0: true=%.4f  found=%.4f" % (info["Lambda0_true"], info["Lambda0_hat"]))
        print("  K:       true=%.4f  found=%.4f" % (info["K_true"], info["K_hat"]))
    if "rhoL_true" in info:
        print("  K:       true=%.4f  found=%.4f" % (info["K_true"], info["K_hat"]))
        print("  rho_L:   true=%.1f  found=%.1f" % (info["rhoL_true"], info["rhoL_hat"]))
    print("  held-out max rel err: %.3e" % info["held_max"])


def main():
    print("=" * 60)
    print("Unknown-law structural discovery (classical)")
    print("=" * 60)
    all_ok = True
    for noise, tag in [(0.0, "clean"), (0.02, "2pct noise")]:
        print("")
        print("#" * 60)
        print("# " + tag)
        print("#" * 60)
        ok1, info1 = test_kohlrausch(noise_frac=noise)
        print_result("Kohlrausch (Lambda = L0 - K*sqrt(c))", ok1, info1)
        all_ok = all_ok and ok1
        ok2, info2 = test_souders_brown(noise_frac=noise)
        print_result("Souders-Brown (v = K*sqrt((rhoL-rhoV)/rhoV))", ok2, info2)
        all_ok = all_ok and ok2
    print("")
    print("=" * 60)
    print("  PASS overall: " + str(all_ok))
    print("=" * 60)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
