"""Benchmark: scaling and ellipsoid tightness on real datasets.

Uses three real datasets:
- Breast Cancer Wisconsin (n=569, d=30) -- sklearn built-in
- German Credit (n=1000, d=61) -- UCI via OpenML, preprocessed at tests/data/
- Adult Census (n=30162, d=104) -- UCI, preprocessed at tests/data/

Run from project root:
    python scripts/benchmark_scale.py
"""
from __future__ import annotations

import csv
import os
import sys
import time

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rashomon import RashomonSet

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "tests", "data")


def load_adult(filepath: str, n_samples: int | None = None) -> tuple:
    data = []
    with open(filepath, encoding="utf-8") as f:
        for row in csv.reader(f):
            if not row:
                continue
            row = [x.strip() for x in row]
            if "?" in row:
                continue
            data.append(row)
            if n_samples and len(data) >= n_samples:
                break
    arr = np.array(data, dtype=object)
    X_raw = arr[:, :-1]
    y = np.array([1.0 if l == ">50K" else 0.0 for l in arr[:, -1]])
    numeric_features = [0, 2, 4, 10, 11, 12]
    categorical_features = [1, 3, 5, 6, 7, 8, 9, 13]
    preprocessor = ColumnTransformer(transformers=[
        ("num", Pipeline([("imp", SimpleImputer(strategy="mean")),
                          ("scl", StandardScaler())]), numeric_features),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="missing")),
                          ("ohe", OneHotEncoder(handle_unknown="ignore",
                                                sparse_output=False))]), categorical_features),
    ])
    X = preprocessor.fit_transform(X_raw)
    return X, y


def load_german_credit() -> tuple:
    path = os.path.join(DATA_DIR, "german_credit.npz")
    if not os.path.exists(path):
        print(f"  German Credit not found at {path}. Fetching from OpenML...")
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
        feature_names = list(pre.get_feature_names_out())
        np.savez_compressed(path, X=X, y=y, feature_names=feature_names)
    data = np.load(path, allow_pickle=True)
    return data["X"], data["y"]


def run_full_pipeline(name: str, X: np.ndarray, y: np.ndarray, C: float = 0.5) -> dict:
    n, d = X.shape
    print(f"\n{'='*60}")
    print(f"{name}: n={n}, d={d}")
    print(f"{'='*60}")

    t0 = time.perf_counter()
    rs = RashomonSet(
        estimator="logistic", C=C, epsilon=0.03,
        epsilon_mode="percent_loss", sampler="ellipsoid",
        random_state=42, safety_override=True,
    ).fit(X, y)
    t_fit = time.perf_counter() - t0
    print(f"  fit:              {t_fit:.3f}s  (L_hat={rs._L_hat:.4f})")

    t0 = time.perf_counter()
    ell = rs.sample_ellipsoid(n_samples=500, random_state=42)
    t_ell = time.perf_counter() - t0
    print(f"  ellipsoid 500:    {t_ell:.3f}s")

    t0 = time.perf_counter()
    hr = rs.sample_hitandrun(n_samples=200, burnin=100, thin=2,
                              random_state=42, compute_diagnostics=True)
    t_hr = time.perf_counter() - t0
    print(f"  hitandrun 200:    {t_hr:.3f}s")

    X_test = X[:min(500, n)]
    t0 = time.perf_counter()
    amb = rs.ambiguity(X_test, threshold_mode="fixed", threshold_value=0.5)
    t_amb = time.perf_counter() - t0
    print(f"  ambiguity ({len(X_test):>3}):  {t_amb:.3f}s  rate={amb['ambiguity_rate']:.1%}")

    t0 = time.perf_counter()
    disc = rs.discrepancy(X_test, n_samples=100, n_pairs=100, random_state=42)
    t_disc = time.perf_counter() - t0
    print(f"  discrepancy:      {t_disc:.3f}s  bound={disc['discrepancy_bound']:.1%}  empirical={disc['max_pair_disagreement']:.1%}")

    t0 = time.perf_counter()
    vic = rs.variable_importance_cloud(n_samples=200, sampler="ellipsoid", random_state=42)
    t_vic = time.perf_counter() - t0
    print(f"  VIC (200):        {t_vic:.3f}s")

    t0 = time.perf_counter()
    cap = rs.capacity(delta=0.01)
    t_cap = time.perf_counter() - t0
    print(f"  capacity:         {t_cap:.3f}s  log_vol={cap['log_volume']:.1f}")

    mem_mb = (X.nbytes + y.nbytes + ell.nbytes + hr.nbytes) / 1e6
    print(f"  est. memory:      {mem_mb:.1f} MB")

    return {"n": n, "d": d, "t_fit": t_fit, "t_ell": t_ell, "t_hr": t_hr,
            "ambiguity": amb["ambiguity_rate"], "discrepancy_bound": disc["discrepancy_bound"]}


def run_tightness(name: str, X: np.ndarray, y: np.ndarray, C: float = 0.5) -> None:
    n, d = X.shape
    print(f"\n{'='*60}")
    print(f"ELLIPSOID TIGHTNESS: {name} (n={n}, d={d})")
    print(f"{'='*60}")

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
        oracle = rs._oracle
        fid = sum(1 for s in ell if oracle.contains(s)) / len(ell)

        r = mc / emp_w if emp_w > 1e-12 else float("inf")
        print(f"  eps={eps:.3f}: cert={mc:.4f}  empirical={emp_w:.4f}  "
              f"ratio={r:.2f}x  fidelity={fid:.0%}")


if __name__ == "__main__":
    # --- Breast Cancer (small, low-d) ---
    bc = load_breast_cancer()
    X_bc = StandardScaler().fit_transform(bc.data)
    y_bc = bc.target.astype(float)
    run_full_pipeline("Breast Cancer", X_bc, y_bc, C=0.5)
    run_tightness("Breast Cancer", X_bc, y_bc, C=0.5)

    # --- German Credit (medium, mid-d) ---
    X_gc, y_gc = load_german_credit()
    run_full_pipeline("German Credit", X_gc, y_gc, C=1.0)
    run_tightness("German Credit", X_gc, y_gc, C=1.0)

    # German Credit diagnostic: C sweep showing degenerate regime
    print(f"\n{'='*60}")
    print("GERMAN CREDIT: C SWEEP (eps=3%)")
    print(f"{'='*60}")
    null_loss = float(np.mean(np.logaddexp(0.0, np.zeros(len(y_gc))) - y_gc * 0.0))
    print(f"  Null loss (theta=0): {null_loss:.4f}")
    for C_val in [0.1, 0.5, 1.0, 5.0]:
        rs_sweep = RashomonSet(
            estimator="logistic", C=C_val, epsilon=0.03,
            epsilon_mode="percent_loss", random_state=42,
            safety_override=True,
        ).fit(X_gc, y_gc)
        acc = rs_sweep.score(X_gc, y_gc)
        amb = rs_sweep.ambiguity(X_gc[:500], threshold_mode="fixed", threshold_value=0.5)
        null_in = rs_sweep._oracle.contains(np.zeros(X_gc.shape[1]))
        print(f"  C={C_val:<4}  acc={acc:.1%}  L_hat={rs_sweep._L_hat:.4f}  "
              f"null_in_set={null_in}  ambiguity={amb['ambiguity_rate']:.1%}")

    # --- Adult Census (large, high-d) ---
    adult_path = os.path.join(DATA_DIR, "adult.data")
    if os.path.exists(adult_path):
        X_ad, y_ad = load_adult(adult_path)
        run_full_pipeline("Adult Census", X_ad, y_ad, C=1.0)
        run_tightness("Adult Census", X_ad, y_ad, C=1.0)

        # Scaling comparison
        print(f"\n{'='*60}")
        print("SCALING (Adult Census, d=104)")
        print(f"{'='*60}")
        for n_sub in [1000, 5000, 10000, len(X_ad)]:
            Xs, ys = X_ad[:n_sub], y_ad[:n_sub]
            t0 = time.perf_counter()
            rs = RashomonSet(estimator="logistic", C=1.0, epsilon=0.03,
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
    else:
        print(f"\nSkipping Adult Census (not found at {adult_path})")

    print("\nDone.")
