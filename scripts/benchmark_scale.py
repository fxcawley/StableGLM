"""Benchmark: scaling, tightness, and certificate calibration on real datasets.

Datasets (all real):
- Breast Cancer PCA-10 (n=569, d=10) -- tight certificates
- Breast Cancer full (n=569, d=30) -- moderate tightness
- German Credit (n=1000, d=61) -- mid-range, credit scoring
- Adult Census (n=30162, d=104) -- large scale

For each dataset, C is selected by 5-fold CV. Both certificate-based
and empirical (Hit-and-Run) ambiguity are reported.

Run: python scripts/benchmark_scale.py
"""
from __future__ import annotations

import csv
import os
import sys
import time

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rashomon import RashomonSet

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "tests", "data")


def load_adult(filepath: str) -> tuple:
    data = []
    with open(filepath, encoding="utf-8") as f:
        for row in csv.reader(f):
            if not row:
                continue
            row = [x.strip() for x in row]
            if "?" in row:
                continue
            data.append(row)
    arr = np.array(data, dtype=object)
    X_raw = arr[:, :-1]
    y = np.array([1.0 if l == ">50K" else 0.0 for l in arr[:, -1]])
    preprocessor = ColumnTransformer(transformers=[
        ("num", Pipeline([("imp", SimpleImputer(strategy="mean")),
                          ("scl", StandardScaler())]), [0, 2, 4, 10, 11, 12]),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="missing")),
                          ("ohe", OneHotEncoder(handle_unknown="ignore",
                                                sparse_output=False))]), [1, 3, 5, 6, 7, 8, 9, 13]),
    ])
    return preprocessor.fit_transform(X_raw), y


def load_german_credit() -> tuple:
    path = os.path.join(DATA_DIR, "german_credit.npz")
    if not os.path.exists(path):
        from sklearn.datasets import fetch_openml
        gc = fetch_openml(data_id=31, as_frame=True, parser="auto")
        X_df = gc.data
        y = (gc.target == "bad").astype(float).values
        num_cols = X_df.select_dtypes(include=["int64", "float64"]).columns.tolist()
        cat_cols = X_df.select_dtypes(include=["object", "category"]).columns.tolist()
        pre = ColumnTransformer(transformers=[
            ("num", Pipeline([("imp", SimpleImputer(strategy="mean")),
                              ("scl", StandardScaler())]), num_cols),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="missing")),
                              ("ohe", OneHotEncoder(handle_unknown="ignore",
                                                    sparse_output=False))]), cat_cols),
        ])
        X = pre.fit_transform(X_df)
        np.savez_compressed(path, X=X, y=y, feature_names=list(pre.get_feature_names_out()))
    d = np.load(path, allow_pickle=True)
    return d["X"], d["y"]


def cv_select_C(X: np.ndarray, y: np.ndarray) -> float:
    """5-fold CV to select C from a grid."""
    grid = GridSearchCV(
        LogisticRegression(fit_intercept=False, solver="lbfgs", l1_ratio=0, max_iter=2000),
        param_grid={"C": [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0]},
        cv=5, scoring="accuracy", n_jobs=1,
    )
    grid.fit(X, y.astype(int))
    return float(grid.best_params_["C"])


def run_dataset(name: str, X: np.ndarray, y: np.ndarray, C: float,
                n_hr: int = 1000, hr_burnin: int = 500, hr_thin: int = 5) -> dict:
    n, d = X.shape
    lam = 1.0 / C
    print(f"\n{'='*70}")
    print(f"{name}: n={n}, d={d}, C={C:.1f} (lambda={lam:.4f})")
    print(f"{'='*70}")

    t0 = time.perf_counter()
    rs = RashomonSet(
        estimator="logistic", C=C, epsilon=0.03,
        epsilon_mode="percent_loss", sampler="hitandrun",
        random_state=42, safety_override=True,
    ).fit(X, y)
    t_fit = time.perf_counter() - t0

    acc = rs.score(X, y)
    null_loss = float(np.mean(np.logaddexp(0.0, np.zeros(n))))
    null_in = rs._oracle.contains(np.zeros(d))
    print(f"  fit: {t_fit:.3f}s  L_hat={rs._L_hat:.4f}  acc={acc:.1%}  null_in_set={null_in}")

    # Certificate ambiguity
    X_test = X[:min(500, n)]
    t0 = time.perf_counter()
    amb_cert = rs.ambiguity(X_test, threshold_mode="fixed", threshold_value=0.5)
    t_cert = time.perf_counter() - t0

    # Empirical ambiguity from HR samples
    t0 = time.perf_counter()
    hr = rs.sample_hitandrun(
        n_samples=n_hr, burnin=hr_burnin, thin=hr_thin,
        random_state=42, compute_diagnostics=True,
    )
    t_hr = time.perf_counter() - t0

    margins = X_test @ hr.T  # (n_test, n_hr)
    preds = (margins > 0.0).astype(int)
    emp_ambiguous = np.any(preds != preds[:, 0:1], axis=1)
    amb_emp = float(np.mean(emp_ambiguous))

    # ESS
    diag = rs.compute_sample_diagnostics(hr)
    ess = diag.get("ess_per_param")
    min_ess = float(np.min(ess)) if ess is not None else float("nan")
    mean_ess = float(np.mean(ess)) if ess is not None else float("nan")

    ratio = amb_cert["ambiguity_rate"] / amb_emp if amb_emp > 0.001 else float("inf")

    print(f"  HR: {n_hr} samples, burnin={hr_burnin}, thin={hr_thin} ({t_hr:.1f}s)")
    print(f"    ESS: min={min_ess:.0f}, mean={mean_ess:.0f}")
    print(f"  Ambiguity (cert.):     {amb_cert['ambiguity_rate']:.1%}  ({t_cert:.3f}s)")
    print(f"  Ambiguity (empirical): {amb_emp:.1%}")
    print(f"  Cert/Emp ratio:        {ratio:.2f}x")

    # Discrepancy
    disc = rs.discrepancy(X_test, samples=hr, n_pairs=200, random_state=42)
    print(f"  Discrepancy: bound={disc['discrepancy_bound']:.1%}  empirical={disc['max_pair_disagreement']:.1%}")

    return {
        "name": name, "n": n, "d": d, "C": C, "acc": acc,
        "L_hat": rs._L_hat, "null_in": null_in,
        "amb_cert": amb_cert["ambiguity_rate"], "amb_emp": amb_emp,
        "cert_emp_ratio": ratio,
        "disc_bound": disc["discrepancy_bound"],
        "disc_emp": disc["max_pair_disagreement"],
        "min_ess": min_ess, "t_hr": t_hr,
    }


def run_tightness(name: str, X: np.ndarray, y: np.ndarray, C: float) -> None:
    n, d = X.shape
    print(f"\n  TIGHTNESS: {name} (d={d})")
    epsilons = [0.005, 0.01, 0.02, 0.03, 0.05, 0.10]
    X_sub = X[:min(n, 2000)]
    y_sub = y[:min(n, 2000)]
    X_test = X_sub[:200]
    for eps in epsilons:
        rs = RashomonSet(
            estimator="logistic", C=C, epsilon=eps,
            epsilon_mode="percent_loss", random_state=42,
            safety_override=True,
        ).fit(X_sub, y_sub)
        cert_w = [rs.hacking_interval(X_test[i])["max"] - rs.hacking_interval(X_test[i])["min"]
                  for i in range(min(100, len(X_test)))]
        mc = float(np.mean(cert_w))
        hr = rs.sample_hitandrun(n_samples=200, burnin=100, thin=2,
                                  random_state=42, compute_diagnostics=False)
        margins = X_test[:100] @ hr.T
        emp_w = float(np.mean(np.max(margins, axis=1) - np.min(margins, axis=1)))
        ell = rs.sample_ellipsoid(n_samples=200, random_state=42)
        fid = sum(1 for s in ell if rs._oracle.contains(s)) / len(ell)
        r = mc / emp_w if emp_w > 1e-12 else float("inf")
        print(f"    eps={eps:.3f}: cert={mc:.4f}  emp={emp_w:.4f}  ratio={r:.2f}x  fidelity={fid:.0%}")


def run_cert_calibration(name: str, X: np.ndarray, y: np.ndarray, C: float,
                          n_hr: int = 500) -> None:
    """Certificate vs empirical ambiguity across epsilon values."""
    n, d = X.shape
    print(f"\n  CERTIFICATE CALIBRATION: {name} (d={d})")
    epsilons = [0.005, 0.01, 0.02, 0.03, 0.05, 0.10]
    X_test = X[:min(500, n)]
    print(f"    {'eps':>6} {'Cert Amb':>10} {'Emp Amb':>10} {'Ratio':>8}")
    print(f"    {'-'*38}")
    for eps in epsilons:
        rs = RashomonSet(
            estimator="logistic", C=C, epsilon=eps,
            epsilon_mode="percent_loss", random_state=42,
            safety_override=True,
        ).fit(X[:min(n, 2000)], y[:min(n, 2000)])
        amb_c = rs.ambiguity(X_test[:min(200, len(X_test))], threshold_mode="fixed", threshold_value=0.5)
        hr = rs.sample_hitandrun(n_samples=n_hr, burnin=200, thin=3,
                                  random_state=42, compute_diagnostics=False)
        margins = X_test[:min(200, len(X_test))] @ hr.T
        preds = (margins > 0.0).astype(int)
        emp_amb = float(np.mean(np.any(preds != preds[:, 0:1], axis=1)))
        r = amb_c["ambiguity_rate"] / emp_amb if emp_amb > 0.001 else float("inf")
        print(f"    {eps:>6.3f} {amb_c['ambiguity_rate']:>9.1%} {emp_amb:>9.1%} {r:>7.2f}x")


if __name__ == "__main__":
    results = []

    # --- Breast Cancer PCA-10 (low-d, tight certificates) ---
    bc = load_breast_cancer()
    X_bc10 = StandardScaler().fit_transform(
        PCA(n_components=10, random_state=42).fit_transform(bc.data)
    )
    y_bc = bc.target.astype(float)
    print("CV for Breast Cancer PCA-10...")
    C_bc10 = cv_select_C(X_bc10, y_bc)
    print(f"  Best C={C_bc10}")
    results.append(run_dataset("Breast Cancer PCA-10", X_bc10, y_bc, C_bc10, n_hr=1000))
    run_tightness("Breast Cancer PCA-10", X_bc10, y_bc, C_bc10)
    run_cert_calibration("Breast Cancer PCA-10", X_bc10, y_bc, C_bc10)

    # --- Breast Cancer full (mid-d) ---
    X_bc30 = StandardScaler().fit_transform(bc.data)
    print("\nCV for Breast Cancer full...")
    C_bc30 = cv_select_C(X_bc30, y_bc)
    print(f"  Best C={C_bc30}")
    results.append(run_dataset("Breast Cancer Full", X_bc30, y_bc, C_bc30, n_hr=1000))
    run_tightness("Breast Cancer Full", X_bc30, y_bc, C_bc30)
    run_cert_calibration("Breast Cancer Full", X_bc30, y_bc, C_bc30)

    # --- German Credit (mid-d, credit scoring) ---
    X_gc, y_gc = load_german_credit()
    print("\nCV for German Credit...")
    C_gc = cv_select_C(X_gc, y_gc)
    print(f"  Best C={C_gc}")
    results.append(run_dataset("German Credit", X_gc, y_gc, C_gc, n_hr=1000))
    run_tightness("German Credit", X_gc, y_gc, C_gc)
    run_cert_calibration("German Credit", X_gc, y_gc, C_gc)

    # --- Adult Census (large, high-d) ---
    adult_path = os.path.join(DATA_DIR, "adult.data")
    if os.path.exists(adult_path):
        X_ad, y_ad = load_adult(adult_path)
        print("\nCV for Adult (subsample 5000 for speed)...")
        C_ad = cv_select_C(X_ad[:5000], y_ad[:5000])
        print(f"  Best C={C_ad}")
        results.append(run_dataset("Adult Census", X_ad, y_ad, C_ad,
                                    n_hr=500, hr_burnin=200, hr_thin=3))
        run_tightness("Adult Census", X_ad, y_ad, C_ad)
        run_cert_calibration("Adult Census", X_ad, y_ad, C_ad, n_hr=300)

        # Scaling
        print(f"\n{'='*70}")
        print("SCALING (Adult Census, d=104)")
        print(f"{'='*70}")
        for n_sub in [1000, 5000, 10000, len(X_ad)]:
            Xs, ys = X_ad[:n_sub], y_ad[:n_sub]
            t0 = time.perf_counter()
            rs = RashomonSet(estimator="logistic", C=C_ad, epsilon=0.03,
                             epsilon_mode="percent_loss", random_state=42,
                             safety_override=True).fit(Xs, ys)
            tf = time.perf_counter() - t0
            t0 = time.perf_counter()
            rs.sample_ellipsoid(n_samples=200, random_state=42)
            te = time.perf_counter() - t0
            t0 = time.perf_counter()
            rs.sample_hitandrun(n_samples=50, burnin=50, thin=1,
                                random_state=42, compute_diagnostics=False)
            th = time.perf_counter() - t0
            print(f"  n={n_sub:>6d}: fit={tf:.3f}s  ell200={te:.3f}s  hr50={th:.3f}s")

    # Summary table
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Dataset':<22} {'n':>6} {'d':>4} {'C(CV)':>6} {'Acc':>6} "
          f"{'Cert.Amb':>9} {'Emp.Amb':>8} {'C/E':>5} {'ESS':>5}")
    print("-" * 75)
    for r in results:
        print(f"{r['name']:<22} {r['n']:>6} {r['d']:>4} {r['C']:>6.1f} {r['acc']:>5.1%} "
              f"{r['amb_cert']:>8.1%} {r['amb_emp']:>7.1%} {r['cert_emp_ratio']:>5.1f}x "
              f"{r['min_ess']:>5.0f}")

    print("\nDone.")
