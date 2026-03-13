"""Benchmark: scaling experiments and ellipsoid tightness analysis.

Generates results for the docs evaluation section. Run from project root:
    python scripts/benchmark_scale.py
"""
from __future__ import annotations

import csv
import os
import sys
import time

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rashomon import RashomonSet


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


def run_scale_experiment(X: np.ndarray, y: np.ndarray) -> dict:
    """Run full pipeline and report timing."""
    n, d = X.shape
    print(f"\n{'='*60}")
    print(f"SCALE EXPERIMENT: n={n}, d={d}")
    print(f"{'='*60}")

    t0 = time.perf_counter()
    rs = RashomonSet(
        estimator="logistic", C=0.01, epsilon=0.03,
        epsilon_mode="percent_loss", sampler="ellipsoid",
        random_state=42, safety_override=True,
    ).fit(X, y)
    t_fit = time.perf_counter() - t0
    print(f"  fit:              {t_fit:.3f}s")

    t0 = time.perf_counter()
    diag = rs.diagnostics()
    t_diag = time.perf_counter() - t0
    print(f"  diagnostics:      {t_diag:.3f}s")
    print(f"    L_hat={diag['L_hat']:.6f}, cond(H)={diag.get('hessian_condition_number', 'N/A')}")

    t0 = time.perf_counter()
    ell_samples = rs.sample_ellipsoid(n_samples=500, random_state=42)
    t_ell = time.perf_counter() - t0
    print(f"  ellipsoid 500:    {t_ell:.3f}s")

    t0 = time.perf_counter()
    hr_samples = rs.sample_hitandrun(n_samples=200, burnin=100, thin=2,
                                      random_state=42, compute_diagnostics=True)
    t_hr = time.perf_counter() - t0
    print(f"  hitandrun 200:    {t_hr:.3f}s")

    X_test = X[:500]
    t0 = time.perf_counter()
    amb = rs.ambiguity(X_test, threshold_mode="fixed", threshold_value=0.5)
    t_amb = time.perf_counter() - t0
    print(f"  ambiguity (500):  {t_amb:.3f}s  rate={amb['ambiguity_rate']:.1%}")

    t0 = time.perf_counter()
    disc = rs.discrepancy(X_test, n_samples=100, n_pairs=100, random_state=42)
    t_disc = time.perf_counter() - t0
    print(f"  discrepancy:      {t_disc:.3f}s  bound={disc['discrepancy_bound']:.1%}")

    t0 = time.perf_counter()
    vic = rs.variable_importance_cloud(n_samples=200, sampler="ellipsoid", random_state=42)
    t_vic = time.perf_counter() - t0
    print(f"  VIC (200):        {t_vic:.3f}s")

    t0 = time.perf_counter()
    cap = rs.capacity(delta=0.01)
    t_cap = time.perf_counter() - t0
    print(f"  capacity:         {t_cap:.3f}s  log_vol={cap['log_volume']:.1f}")

    mem_mb = (X.nbytes + y.nbytes + ell_samples.nbytes + hr_samples.nbytes) / 1e6
    print(f"  est. memory:      {mem_mb:.1f} MB (data + samples)")

    return {
        "n": n, "d": d,
        "t_fit": t_fit, "t_ellipsoid": t_ell, "t_hitandrun": t_hr,
        "t_ambiguity": t_amb, "t_discrepancy": t_disc, "t_vic": t_vic,
        "ambiguity_rate": amb["ambiguity_rate"],
        "discrepancy_bound": disc["discrepancy_bound"],
        "L_hat": diag["L_hat"],
    }


def run_tightness_analysis(X: np.ndarray, y: np.ndarray) -> list[dict]:
    """Compare ellipsoidal certificate widths to Hit-and-Run empirical widths."""
    n, d = X.shape
    print(f"\n{'='*60}")
    print(f"ELLIPSOID TIGHTNESS ANALYSIS: n={n}, d={d}")
    print(f"{'='*60}")

    epsilons = [0.005, 0.01, 0.02, 0.03, 0.05, 0.10]
    results = []

    # Use a subset for speed
    X_sub = X[:min(n, 2000)]
    y_sub = y[:min(n, 2000)]
    X_test = X_sub[:200]

    for eps in epsilons:
        rs = RashomonSet(
            estimator="logistic", C=0.01, epsilon=eps,
            epsilon_mode="percent_loss", random_state=42,
            safety_override=True,
        ).fit(X_sub, y_sub)

        # Certificate widths (ellipsoidal)
        cert_widths = []
        for i in range(min(100, len(X_test))):
            interval = rs.hacking_interval(X_test[i])
            cert_widths.append(interval["max"] - interval["min"])
        mean_cert = float(np.mean(cert_widths))

        # Empirical widths from Hit-and-Run
        hr_samples = rs.sample_hitandrun(
            n_samples=200, burnin=100, thin=2,
            random_state=42, compute_diagnostics=False,
        )
        # Compute margin range for each test instance across HR samples
        margins = X_test[:100] @ hr_samples.T
        empirical_widths = np.max(margins, axis=1) - np.min(margins, axis=1)
        mean_empirical = float(np.mean(empirical_widths))

        # Ellipsoid set fidelity
        ell_samples = rs.sample_ellipsoid(n_samples=200, random_state=42)
        oracle = rs._oracle
        inside = sum(1 for s in ell_samples if oracle.contains(s))
        fidelity = inside / len(ell_samples)

        ratio = mean_cert / mean_empirical if mean_empirical > 1e-12 else float("inf")
        print(f"  eps={eps:.3f}: cert={mean_cert:.4f}  empirical={mean_empirical:.4f}  "
              f"ratio={ratio:.2f}x  fidelity={fidelity:.0%}")

        results.append({
            "epsilon": eps,
            "mean_cert_width": mean_cert,
            "mean_empirical_width": mean_empirical,
            "cert_to_empirical_ratio": ratio,
            "ellipsoid_fidelity": fidelity,
        })

    return results


if __name__ == "__main__":
    data_path = os.path.join(os.path.dirname(__file__), "..", "tests", "data", "adult.data")
    if not os.path.exists(data_path):
        print(f"Adult dataset not found at {data_path}")
        sys.exit(1)

    # Full-scale Adult
    X_full, y_full = load_adult(data_path)
    scale_results = run_scale_experiment(X_full, y_full)

    # Tightness on a manageable subset
    tightness_results = run_tightness_analysis(X_full, y_full)

    # Also run at smaller scales for comparison
    print(f"\n{'='*60}")
    print("SCALING COMPARISON")
    print(f"{'='*60}")
    for n_sub in [1000, 5000, 10000, len(X_full)]:
        X_s, y_s = X_full[:n_sub], y_full[:n_sub]
        t0 = time.perf_counter()
        rs = RashomonSet(
            estimator="logistic", C=0.01, epsilon=0.03,
            epsilon_mode="percent_loss", random_state=42,
            safety_override=True,
        ).fit(X_s, y_s)
        t_fit = time.perf_counter() - t0
        t0 = time.perf_counter()
        rs.sample_ellipsoid(n_samples=200, random_state=42)
        t_ell = time.perf_counter() - t0
        t0 = time.perf_counter()
        rs.sample_hitandrun(n_samples=50, burnin=50, thin=1,
                            random_state=42, compute_diagnostics=False)
        t_hr = time.perf_counter() - t0
        print(f"  n={n_sub:>6d}, d={X_s.shape[1]:>3d}: "
              f"fit={t_fit:.3f}s  ell200={t_ell:.3f}s  hr50={t_hr:.3f}s")

    print("\nDone.")
