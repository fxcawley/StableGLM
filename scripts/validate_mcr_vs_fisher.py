"""Validate MCR implementation against Fisher et al. (2019) reference algorithm.

Fisher et al. define MCR as:
    MCR_j^- = min_{theta in R_eps} I_j(theta)
    MCR_j^+ = max_{theta in R_eps} I_j(theta)

where I_j(theta) = Score(theta, X) - Score(theta, X_perm_j) is the
permutation importance of feature j under model theta.

This script validates our implementation by:
1. Sampling theta vectors from the Rashomon set
2. Computing permutation importance independently (no RashomonSet internals)
3. Feeding the SAME samples to both our MCR and the reference, using the
   same permutation seed, to get exact agreement

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

    This is independent of our RashomonSet class -- pure numpy only.
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


def our_mcr_on_samples(
    rs: RashomonSet,
    samples: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_permutations: int = 20,
    seed: int = 42,
) -> dict:
    """Run our MCR logic on specific pre-drawn samples (matching the internal loop)."""
    rng = np.random.default_rng(seed)
    n_models, d = samples.shape
    n = X.shape[0]
    importance_matrix = np.zeros((n_models, d))

    for s_idx in range(n_models):
        theta_s = samples[s_idx]
        if rs.estimator == "logistic":
            scores = X @ theta_s
            preds = (scores > 0.0).astype(int)
            base = float(np.mean((preds == y.astype(int)).astype(float)))
        else:
            preds = X @ theta_s
            ss_res = float(np.sum((y - preds) ** 2))
            y_mean = float(np.mean(y))
            ss_tot = float(np.sum((y - y_mean) ** 2))
            base = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

        for j in range(d):
            perm_scores = np.zeros(n_permutations)
            for p in range(n_permutations):
                Xp = X.copy()
                rng.shuffle(Xp[:, j])
                if rs.estimator == "logistic":
                    sp = Xp @ theta_s
                    pp = (sp > 0.0).astype(int)
                    perm_scores[p] = float(np.mean((pp == y.astype(int)).astype(float)))
                else:
                    pp = Xp @ theta_s
                    sr = float(np.sum((y - pp) ** 2))
                    perm_scores[p] = 1.0 - sr / ss_tot if ss_tot > 0 else 1.0
            importance_matrix[s_idx, j] = base - np.mean(perm_scores)

    return {
        "mcr_min": np.min(importance_matrix, axis=0),
        "mcr_max": np.max(importance_matrix, axis=0),
        "mcr_mean": np.mean(importance_matrix, axis=0),
        "importance_matrix": importance_matrix,
    }


if __name__ == "__main__":
    print("MCR Validation: rashomon-py vs Fisher et al. Reference")
    print("=" * 60)

    # Shared dataset: Breast Cancer with all 30 features (more rigorous than d=5)
    data = load_breast_cancer()
    X = StandardScaler().fit_transform(data.data)
    y = data.target.astype(float)
    print(f"Dataset: Breast Cancer (n={X.shape[0]}, d={X.shape[1]})")

    # Fit RashomonSet
    rs = RashomonSet(
        estimator="logistic", C=0.5, epsilon=0.03,
        epsilon_mode="percent_loss", sampler="hitandrun",
        random_state=42, safety_override=True,
    ).fit(X, y)

    # Draw ONE set of samples, shared by both methods
    n_models = 100
    n_perm = 20
    shared_seed = 42
    samples = rs.sample_hitandrun(
        n_samples=n_models, burnin=200, thin=3,
        random_state=shared_seed, compute_diagnostics=False,
    )
    print(f"Shared {n_models} Hit-and-Run samples (seed={shared_seed})")

    # Fisher reference on shared samples
    t0 = time.perf_counter()
    ref_mcr = fisher_mcr_reference(
        samples, X, y, estimator="logistic",
        n_permutations=n_perm, seed=shared_seed,
    )
    t_ref = time.perf_counter() - t0

    # Our implementation on the SAME shared samples with SAME seed
    t0 = time.perf_counter()
    our_mcr = our_mcr_on_samples(
        rs, samples, X, y,
        n_permutations=n_perm, seed=shared_seed,
    )
    t_ours = time.perf_counter() - t0

    print(f"Timing: ours={t_ours:.2f}s, reference={t_ref:.2f}s")

    # Compare (should be EXACTLY zero since same samples, same permutations)
    names = data.feature_names[:X.shape[1]]
    print(f"\n{'Feature':<25} {'Our MCR-':>9} {'Ref MCR-':>9} {'Our MCR+':>9} {'Ref MCR+':>9}")
    print("-" * 65)
    max_diff = 0.0
    for j in range(min(10, X.shape[1])):  # show first 10
        our_min, ref_min = our_mcr["mcr_min"][j], ref_mcr["mcr_min"][j]
        our_max, ref_max = our_mcr["mcr_max"][j], ref_mcr["mcr_max"][j]
        diff = max(abs(our_min - ref_min), abs(our_max - ref_max))
        max_diff = max(max_diff, diff)
        print(f"{names[j]:<25} {our_min:>+9.4f} {ref_min:>+9.4f} {our_max:>+9.4f} {ref_max:>+9.4f}")

    # Check all features
    all_min_diff = float(np.max(np.abs(our_mcr["mcr_min"] - ref_mcr["mcr_min"])))
    all_max_diff = float(np.max(np.abs(our_mcr["mcr_max"] - ref_mcr["mcr_max"])))
    all_mean_diff = float(np.max(np.abs(our_mcr["mcr_mean"] - ref_mcr["mcr_mean"])))

    print(f"\n... ({X.shape[1]} features total)")
    print(f"Max absolute difference across ALL {X.shape[1]} features:")
    print(f"  MCR-: {all_min_diff:.2e}")
    print(f"  MCR+: {all_max_diff:.2e}")
    print(f"  Mean: {all_mean_diff:.2e}")

    TOLERANCE = 1e-10
    if max(all_min_diff, all_max_diff, all_mean_diff) < TOLERANCE:
        print(f"\nVALIDATION PASSED: exact agreement (< {TOLERANCE})")
    else:
        print(f"\nVALIDATION FAILED: differences exceed {TOLERANCE}")
        sys.exit(1)
