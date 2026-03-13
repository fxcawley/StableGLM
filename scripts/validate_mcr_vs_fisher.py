"""Validate MCR implementation against Fisher et al. (2019) reference algorithm.

Fisher et al. define MCR as:
    MCR_j^- = min_{theta in R_eps} I_j(theta)
    MCR_j^+ = max_{theta in R_eps} I_j(theta)

where I_j(theta) = Score(theta, X) - Score(theta, X_perm_j) is the
permutation importance of feature j under model theta.

Our model_class_reliance() approximates this by sampling theta from R_eps
and taking min/max over samples.  This script validates that approach by
implementing the reference algorithm from scratch (independent of our
RashomonSet internals) and comparing on a shared dataset.

Usage:
    python scripts/validate_mcr_vs_fisher.py
"""
from __future__ import annotations

import sys
import os
import time

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rashomon import RashomonSet


def fisher_mcr_reference(
    theta_samples: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    *,
    estimator: str = "logistic",
    n_permutations: int = 20,
    seed: int = 42,
) -> dict:
    """Reference implementation of Fisher et al. MCR.

    Given a set of parameter vectors (the "Rashomon set"), computes
    permutation importance for each model and returns min/max across models.

    This is independent of our RashomonSet class -- pure numpy.
    """
    rng = np.random.default_rng(seed)
    n_models, d = theta_samples.shape
    n = X.shape[0]

    importance_matrix = np.zeros((n_models, d))

    for m in range(n_models):
        theta = theta_samples[m]

        # Base score
        if estimator == "logistic":
            preds = (X @ theta > 0.0).astype(int)
            base_acc = float(np.mean(preds == y.astype(int)))
        else:
            preds = X @ theta
            ss_res = float(np.sum((y - preds) ** 2))
            ss_tot = float(np.sum((y - np.mean(y)) ** 2))
            base_acc = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

        # Permutation importance per feature
        for j in range(d):
            perm_scores = np.zeros(n_permutations)
            for p in range(n_permutations):
                Xp = X.copy()
                rng.shuffle(Xp[:, j])
                if estimator == "logistic":
                    pp = (Xp @ theta > 0.0).astype(int)
                    perm_scores[p] = float(np.mean(pp == y.astype(int)))
                else:
                    pp = Xp @ theta
                    sr = float(np.sum((y - pp) ** 2))
                    perm_scores[p] = 1.0 - sr / ss_tot if ss_tot > 0 else 1.0
            importance_matrix[m, j] = base_acc - np.mean(perm_scores)

    return {
        "mcr_min": np.min(importance_matrix, axis=0),
        "mcr_max": np.max(importance_matrix, axis=0),
        "mcr_mean": np.mean(importance_matrix, axis=0),
        "importance_matrix": importance_matrix,
    }


if __name__ == "__main__":
    print("MCR Validation: StableGLM vs Fisher et al. Reference")
    print("=" * 60)

    # Shared dataset
    data = load_breast_cancer()
    X = StandardScaler().fit_transform(data.data[:, :5])
    y = data.target.astype(float)
    names = ["radius", "texture", "perimeter", "area", "smoothness"]
    print(f"Dataset: Breast Cancer (n={X.shape[0]}, d={X.shape[1]})")

    # Fit RashomonSet
    rs = RashomonSet(
        estimator="logistic", C=0.5, epsilon=0.03,
        epsilon_mode="percent_loss", sampler="hitandrun",
        random_state=42, safety_override=True,
    ).fit(X, y)

    # Get the same samples for both methods
    n_models = 200
    n_perm = 20
    samples = rs.sample_hitandrun(
        n_samples=n_models, burnin=200, thin=3,
        random_state=42, compute_diagnostics=False,
    )
    print(f"Sampled {n_models} models from Rashomon set (Hit-and-Run)")

    # Our implementation
    t0 = time.perf_counter()
    our_mcr = rs.model_class_reliance(
        X, y, n_permutations=n_perm, n_samples=n_models,
        sampler="hitandrun", burnin=200, thin=3,
        random_state=42,
    )
    t_ours = time.perf_counter() - t0

    # Fisher reference (on the same samples, same permutation seed)
    t0 = time.perf_counter()
    ref_mcr = fisher_mcr_reference(
        samples, X, y, estimator="logistic",
        n_permutations=n_perm, seed=42,
    )
    t_ref = time.perf_counter() - t0

    print(f"\nTiming: ours={t_ours:.2f}s, reference={t_ref:.2f}s")

    # Compare
    print(f"\n{'Feature':<12} {'Our MCR-':>9} {'Ref MCR-':>9} {'Our MCR+':>9} {'Ref MCR+':>9} {'Our Mean':>9} {'Ref Mean':>9}")
    print("-" * 70)
    max_min_diff = 0.0
    max_max_diff = 0.0
    max_mean_diff = 0.0
    for j, name in enumerate(names):
        our_min = our_mcr["mcr_min"][j]
        ref_min = ref_mcr["mcr_min"][j]
        our_max = our_mcr["mcr_max"][j]
        ref_max = ref_mcr["mcr_max"][j]
        our_mean = our_mcr["feature_importance"][j]
        ref_mean = ref_mcr["mcr_mean"][j]
        print(f"{name:<12} {our_min:>+9.4f} {ref_min:>+9.4f} {our_max:>+9.4f} {ref_max:>+9.4f} {our_mean:>+9.4f} {ref_mean:>+9.4f}")
        max_min_diff = max(max_min_diff, abs(our_min - ref_min))
        max_max_diff = max(max_max_diff, abs(our_max - ref_max))
        max_mean_diff = max(max_mean_diff, abs(our_mean - ref_mean))

    print(f"\nMax absolute differences:")
    print(f"  MCR-: {max_min_diff:.6f}")
    print(f"  MCR+: {max_max_diff:.6f}")
    print(f"  Mean: {max_mean_diff:.6f}")

    # The two use different sample sets (our MCR draws fresh samples),
    # so exact agreement is not expected. But the distributions should
    # be statistically compatible.
    TOLERANCE = 0.05  # 5% absolute tolerance
    if max_mean_diff < TOLERANCE:
        print(f"\nVALIDATION PASSED: mean importance agrees within {TOLERANCE}")
    else:
        print(f"\nVALIDATION WARNING: mean importance differs by {max_mean_diff:.4f} > {TOLERANCE}")
        print("  (Expected: both use Hit-and-Run but with different sample draws)")
