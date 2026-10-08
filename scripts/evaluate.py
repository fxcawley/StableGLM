"""Evaluation on real datasets: exact quantities vs. sampled and ellipsoid approximations.

For each dataset a scikit-learn LogisticRegression is fitted with a cross-validated C
on standardized features and audited with ``audit()`` (so the regularization strength
is converted correctly). The exact flip rate and exact coefficient ranges are then
compared with what the sampler and the Hessian-ellipsoid approximation give.

Datasets (all real; Adult is subsampled so the exact computations finish in minutes):
- Breast Cancer PCA-10 (n=569, d=10)
- Breast Cancer full   (n=569, d=30)
- German Credit        (n=1000, d=61)  tests/data/german_credit.npz
- Adult Census         (n=5000 of 30162, d=104)  tests/data/adult.data (skipped if absent)

Run from the repository root:  python scripts/evaluate.py [--quick]
Writes docs/_static/evaluation.json and prints the Markdown tables used in
docs/evaluation.md.
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from benchmark_scale import load_adult, load_german_credit  # noqa: E402
from rashomon import audit  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "tests", "data")
QUICK = "--quick" in sys.argv
N_SAMPLES = 500 if QUICK else 2000
TOLERANCES = ["cv", 0.01, 0.03]


def select_c(X: np.ndarray, y: np.ndarray) -> float:
    grid = GridSearchCV(
        LogisticRegression(max_iter=5000),
        param_grid={"C": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]},
        cv=StratifiedKFold(5, shuffle=True, random_state=0),
        scoring="neg_log_loss",
    )
    grid.fit(X, y)
    return float(grid.best_params_["C"])


def evaluate(name: str, X: np.ndarray, y: np.ndarray) -> list[dict]:
    X = StandardScaler().fit_transform(X)
    n, d = X.shape
    C = select_c(X, y)
    model = LogisticRegression(C=C, max_iter=5000, tol=1e-8).fit(X, y)
    acc = float(model.score(X, y))
    rows = []
    for tol in TOLERANCES:
        t0 = time.perf_counter()
        exact = audit(model, X, y, tolerance=tol, n_samples=N_SAMPLES, exact_ranges=True, exact_flips=True, random_state=0)
        t_exact = time.perf_counter() - t0
        sampled = audit(model, X, y, tolerance=("absolute", exact.tolerance), n_samples=N_SAMPLES,
                        exact_ranges=False, exact_flips=False, random_state=0)
        rs = exact.rashomon_set
        # ellipsoid approximation of the coefficient ranges vs the exact ranges
        offset = 1 if rs.fit_intercept else 0
        ell = rs.coef_intervals()[offset:]
        ex = np.c_[exact.coefficients["low"].to_numpy(), exact.coefficients["high"].to_numpy()]
        ell_ratio = (ell[:, 1] - ell[:, 0]) / (ex[:, 1] - ex[:, 0])
        smp = np.c_[sampled.coefficients["low"].to_numpy(), sampled.coefficients["high"].to_numpy()]
        smp_cover = (smp[:, 1] - smp[:, 0]) / (ex[:, 1] - ex[:, 0])
        # ellipsoid flip screen: rows whose ellipsoid margin interval straddles the threshold
        Xa = rs._prepare_X(X)
        ell_flip = np.array([rs.hacking_interval(Xa[i])["min"] <= 0.0 <= rs.hacking_interval(Xa[i])["max"] for i in range(n)])
        rows.append({
            "dataset": name, "n": n, "d": d, "C_sklearn": C, "accuracy": acc,
            "tolerance": tol if isinstance(tol, str) else f"{tol:g}", "epsilon": exact.tolerance,
            "epsilon_over_loss": exact.tolerance / exact.loss_optimum,
            "flip_exact": exact.flip_rate, "flip_sampled": sampled.flip_rate, "flip_ellipsoid": float(ell_flip.mean()),
            "sign_stable_exact": int(exact.coefficients["sign_stable"].sum()),
            "sign_stable_sampled": int(sampled.coefficients["sign_stable"].sum()),
            "sampled_range_coverage_median": float(np.median(smp_cover)),
            "ellipsoid_range_ratio_median": float(np.median(ell_ratio)),
            "ellipsoid_range_ratio_max": float(np.max(ell_ratio)),
            "ess_min": exact.ess_min, "seconds_exact": t_exact,
        })
        print(f"  {name:22} tol={rows[-1]['tolerance']:>5} eps={exact.tolerance:.4g} ({rows[-1]['epsilon_over_loss']:.1%} of loss) "
              f"flips exact={exact.flip_rate:.1%} sampled={sampled.flip_rate:.1%} ellipsoid={ell_flip.mean():.1%} | "
              f"sign-stable exact={rows[-1]['sign_stable_exact']}/{d} sampled={rows[-1]['sign_stable_sampled']}/{d} | "
              f"sampled range coverage={np.median(smp_cover):.0%} ellipsoid/exact width={np.median(ell_ratio):.2f}x "
              f"(max {np.max(ell_ratio):.2f}x) | ESS={exact.ess_min:.0f} | {t_exact:.0f}s", flush=True)
    return rows


def markdown(results: list[dict]) -> str:
    out = ["| Dataset | n | d | C | tolerance | ε / loss | flip rate: exact | sampled | ellipsoid | sign-stable: exact | sampled | sampled range coverage | ellipsoid / exact width | min ESS |",
           "|---|--:|--:|--:|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for r in results:
        out.append(
            f"| {r['dataset']} | {r['n']} | {r['d']} | {r['C_sklearn']:g} | {r['tolerance']} | {r['epsilon_over_loss']:.1%} | "
            f"{r['flip_exact']:.1%} | {r['flip_sampled']:.1%} | {r['flip_ellipsoid']:.1%} | {r['sign_stable_exact']}/{r['d']} | "
            f"{r['sign_stable_sampled']}/{r['d']} | {r['sampled_range_coverage_median']:.0%} | "
            f"{r['ellipsoid_range_ratio_median']:.2f}x (max {r['ellipsoid_range_ratio_max']:.2f}x) | {r['ess_min']:.0f} |"
        )
    return "\n".join(out)


if __name__ == "__main__":
    results: list[dict] = []
    bc = load_breast_cancer()
    y_bc = bc.target.astype(float)
    results += evaluate("Breast Cancer PCA-10", PCA(n_components=10, random_state=0).fit_transform(StandardScaler().fit_transform(bc.data)), y_bc)
    results += evaluate("Breast Cancer full", bc.data, y_bc)
    X_gc, y_gc = load_german_credit()
    results += evaluate("German Credit", X_gc, y_gc)
    adult_path = os.path.join(DATA_DIR, "adult.data")
    if os.path.exists(adult_path):
        X_ad, y_ad = load_adult(adult_path)
        rng = np.random.default_rng(0)
        idx = rng.choice(X_ad.shape[0], size=5000, replace=False)
        X_ad, y_ad = X_ad[idx], y_ad[idx]
        keep = X_ad.std(axis=0) > 0  # drop one-hot columns that are empty in the subsample
        results += evaluate("Adult Census (5k rows)", X_ad[:, keep], y_ad)
    else:
        print("adult.data not found; skipping Adult Census")
    out = os.path.join(os.path.dirname(__file__), "..", "docs", "_static", "evaluation.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("\n" + markdown(results))
    print(f"\nwrote {out}")
