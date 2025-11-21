"""Comprehensive demonstration of StableGLM/rashomon-py on real-world data.

This script demonstrates the full pipeline:
1. Data loading and preprocessing
2. Model fitting with regularization
3. Diagnostics and quality checks
4. Sampling methods (Ellipsoid and Hit-and-Run)
5. VIC (Variable Importance Cloud)
6. MCR (Model Class Reliance) with collinearity detection
7. Predictive multiplicity metrics (ambiguity, discrepancy)
8. Visualization (if matplotlib available)
"""

import time

import numpy as np

from rashomon import RashomonSet


def print_header(title):
    """Print formatted section header."""
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print('=' * 80)


def main():
    """Run comprehensive demonstration."""
    print_header("StableGLM/rashomon-py: Comprehensive Demonstration")
    print("\nThis demo showcases all key features on a real-world dataset.")
    
    # =========================================================================
    # 1. DATA PREPARATION
    # =========================================================================
    print_header("1. Data Preparation")
    
    try:
        from sklearn.datasets import load_breast_cancer
        from sklearn.decomposition import PCA
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        
        # Load data
        data = load_breast_cancer()
        print(f"Dataset: Wisconsin Breast Cancer")
        print(f"  Samples: {data.data.shape[0]}")
        print(f"  Features: {data.data.shape[1]}")
        print(f"  Classes: {np.unique(data.target)}")
        print(f"  Prevalence: {np.mean(data.target):.2%}")
        
        # Preprocess with PCA to avoid separation
        X_full = StandardScaler().fit_transform(data.data)
        pca = PCA(n_components=15, random_state=42)
        X = pca.fit_transform(X_full)
        y = data.target.astype(float)
        
        print(f"\nAfter PCA: {X.shape[1]} components (explained variance: {pca.explained_variance_ratio_.sum():.2%})")
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print(f"Train: {len(X_train)} samples, Test: {len(X_test)} samples")
        
        feature_names = [f"PC{i+1}" for i in range(X.shape[1])]
        
    except ImportError:
        print("sklearn not available - using synthetic data")
        np.random.seed(42)
        n, d = 300, 10
        X = np.random.randn(n, d)
        true_coef = np.random.randn(d)
        y = (X @ true_coef + np.random.randn(n) > 0).astype(float)
        X_train, X_test = X[:240], X[240:]
        y_train, y_test = y[:240], y[240:]
        feature_names = [f"X{i}" for i in range(d)]
    
    # =========================================================================
    # 2. MODEL FITTING
    # =========================================================================
    print_header("2. Model Fitting")
    
    start_time = time.time()
    rs = RashomonSet(
        estimator="logistic",
        C=0.01,  # Strong L2 regularization
        epsilon_mode="percent_loss",
        epsilon=0.05,  # 5% loss tolerance
        safety_override=True,  # Override separation checks for clean data
        random_state=42
    )
    rs.fit(X_train, y_train)
    fit_time = time.time() - start_time
    
    print(f"Fit time: {fit_time:.3f}s")
    print(f"Optimal model:")
    print(f"  Train accuracy: {rs.score(X_train, y_train):.2%}")
    print(f"  Test accuracy: {rs.score(X_test, y_test):.2%}")
    
    # =========================================================================
    # 3. DIAGNOSTICS
    # =========================================================================
    print_header("3. Diagnostics & Quality Checks")
    
    diag = rs.diagnostics()
    print(f"Loss at optimal: L_hat = {diag['L_hat']:.4f}")
    print(f"Epsilon value: {diag['epsilon']:.4f} ({rs.epsilon * 100:.1f}% of L_hat)")
    print(f"Regularization: C = {rs.C} (lambda = {1.0/rs.C:.4f})")
    if diag.get('kappa_H'):
        print(f"Condition number: {diag['kappa_H']:.2e}")
    
    # Coefficient intervals
    intervals = rs.coef_intervals()
    n_certain = np.sum((intervals[:, 0] > 0) | (intervals[:, 1] < 0))
    print(f"\nCoefficients with sign certainty: {n_certain}/{len(intervals)}")
    
    print("\nTop 5 features by |coefficient interval center|:")
    centers = (intervals[:, 0] + intervals[:, 1]) / 2
    top_idx = np.argsort(np.abs(centers))[-5:][::-1]
    for idx in top_idx:
        feat = feature_names[idx] if feature_names else f"X{idx}"
        width = intervals[idx, 1] - intervals[idx, 0]
        print(f"  {feat:15s}: [{intervals[idx, 0]:7.3f}, {intervals[idx, 1]:7.3f}] "
              f"(width: {width:.3f})")
    
    # =========================================================================
    # 4. SAMPLING METHODS
    # =========================================================================
    print_header("4. Sampling from the Rashomon Set")
    
    print("\n--- Ellipsoid Sampler (fast, biased) ---")
    start_time = time.time()
    samples_ell = rs.sample_ellipsoid(n_samples=100, random_state=42)
    ell_time = time.time() - start_time
    diag_ell = rs.compute_sample_diagnostics(samples_ell)
    
    print(f"  Time: {ell_time:.3f}s")
    print(f"  Set fidelity: {diag_ell['set_fidelity']:.1%} (% of samples in Rashomon set)")
    print(f"  Chord mean: {diag_ell['chord_mean']:.4f}")
    print(f"  Chord std: {diag_ell['chord_std']:.4f}")
    
    print("\n--- Hit-and-Run Sampler (slower, unbiased) ---")
    start_time = time.time()
    samples_har = rs.sample_hitandrun(
        n_samples=100,
        burnin=50,
        thin=2,
        random_state=42
    )
    har_time = time.time() - start_time
    diag_har = rs.compute_sample_diagnostics(samples_har)
    
    print(f"  Time: {har_time:.3f}s")
    print(f"  Set fidelity: {diag_har['set_fidelity']:.1%}")
    print(f"  Chord mean: {diag_har['chord_mean']:.4f}")
    print(f"  Chord std: {diag_har['chord_std']:.4f}")
    if diag_har.get('isotropy_ratio'):
        print(f"  Isotropy ratio: {diag_har['isotropy_ratio']:.2f}")
    
    print(f"\nRecommendation: Use Hit-and-Run for unbiased samples ({har_time/ell_time:.1f}x slower)")
    
    # =========================================================================
    # 5. VARIABLE IMPORTANCE CLOUD (VIC)
    # =========================================================================
    print_header("5. Variable Importance Cloud (VIC)")
    
    start_time = time.time()
    vic = rs.variable_importance_cloud(
        n_samples=100,
        sampler="hitandrun",
        feature_names=feature_names,
        random_state=42
    )
    vic_time = time.time() - start_time
    
    print(f"VIC computation time: {vic_time:.3f}s")
    print("\nTop 5 features by importance (mean |coefficient|):")
    top_vic = np.argsort(np.abs(vic['mean']))[-5:][::-1]
    for idx in top_vic:
        feat = feature_names[idx] if feature_names else f"X{idx}"
        ci_width = vic['quantiles'][0.95][idx] - vic['quantiles'][0.05][idx]
        print(f"  {feat:15s}: mean={vic['mean'][idx]:7.3f}, std={vic['std'][idx]:6.3f}, "
              f"90% CI=[{vic['quantiles'][0.05][idx]:7.3f}, {vic['quantiles'][0.95][idx]:7.3f}]")
    
    # =========================================================================
    # 6. MODEL CLASS RELIANCE (MCR)
    # =========================================================================
    print_header("6. Model Class Reliance (MCR)")
    
    start_time = time.time()
    mcr = rs.model_class_reliance(
        X_train, y_train,
        n_samples=50,
        n_permutations=10,
        perm_mode="conditional",  # Best for correlated features
        check_collinearity=True,
        random_state=42
    )
    mcr_time = time.time() - start_time
    
    print(f"MCR computation time: {mcr_time:.3f}s")
    print(f"Permutation mode: conditional (correlation-aware)")
    
    if mcr['collinearity_warning']:
        print(f"\nCollinearity detected: {len(mcr['collinearity_warning'])} pairs")
        for i, j, corr in mcr['collinearity_warning'][:3]:
            feat_i = feature_names[i] if feature_names else f"X{i}"
            feat_j = feature_names[j] if feature_names else f"X{j}"
            print(f"  {feat_i} <-> {feat_j}: r={corr:.3f}")
    
    print("\nTop 5 features by MCR importance:")
    top_mcr = np.argsort(mcr['feature_importance'])[-5:][::-1]
    for idx in top_mcr:
        feat = feature_names[idx] if feature_names else f"X{idx}"
        print(f"  {feat:15s}: importance={mcr['feature_importance'][idx]:.4f} "
              f"(std={mcr['importance_std'][idx]:.4f})")
    
    # Compare VIC and MCR rankings
    print("\n--- Agreement between VIC and MCR ---")
    overlap = len(set(top_vic) & set(top_mcr))
    print(f"Top-5 overlap: {overlap}/5 features")
    
    # =========================================================================
    # 7. PREDICTIVE MULTIPLICITY
    # =========================================================================
    print_header("7. Predictive Multiplicity Metrics")
    
    start_time = time.time()
    mult = rs.multiplicity(
        X_test,
        which=["ambiguity", "discrepancy"],
        y=y_test,
        n_samples=100,
        n_pairs=50,
        threshold_mode="match_prevalence",
        random_state=42
    )
    mult_time = time.time() - start_time
    
    print(f"Multiplicity computation time: {mult_time:.3f}s")
    
    print("\n--- Ambiguity (prediction instability) ---")
    print(f"  Ambiguous instances: {mult['ambiguity']['n_ambiguous']}/{len(X_test)} "
          f"({mult['ambiguity']['ambiguity_rate']:.2%})")
    print(f"  Decision threshold: {mult['ambiguity']['threshold']:.3f}")
    print(f"  Interpretation: {mult['ambiguity']['ambiguity_rate']:.1%} of test instances have "
          f"uncertain predictions")
    
    print("\n--- Discrepancy (maximum disagreement) ---")
    print(f"  Theoretical bound: {mult['discrepancy']['discrepancy_bound']:.2%}")
    print(f"  Empirical estimate: {mult['discrepancy']['discrepancy_empirical']:.2%}")
    print(f"  Max pair disagreement: {mult['discrepancy']['max_pair_disagreement']:.2%}")
    print(f"  Interpretation: Up to {mult['discrepancy']['discrepancy_empirical']:.1%} of "
          f"predictions differ between models")
    
    # Test different threshold modes
    print("\n--- Threshold Sensitivity ---")
    for mode in ["fixed", "match_prevalence"]:
        amb = rs.ambiguity(X_test, threshold_mode=mode, threshold_value=0.5, y=y_test)
        print(f"  {mode:20s}: ambiguity={amb['ambiguity_rate']:5.1%}, "
              f"threshold={amb['threshold']:.3f}")
    
    # =========================================================================
    # 8. VISUALIZATION (optional)
    # =========================================================================
    print_header("8. Visualization")
    
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        
        # Create VIC plot
        fig = rs.plot_vic(
            n_samples=100,
            sampler="hitandrun",
            feature_names=feature_names,
            random_state=42
        )
        vic_plot_path = "vic_plot.png"
        plt.savefig(vic_plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"VIC plot saved to: {vic_plot_path}")
        
    except ImportError:
        print("matplotlib not available - skipping visualization")
    except Exception as e:
        print(f"Visualization failed: {e}")
    
    # =========================================================================
    # 9. SUMMARY
    # =========================================================================
    print_header("9. Summary & Recommendations")
    
    print("\nKey Findings:")
    print(f"  - Test accuracy: {rs.score(X_test, y_test):.2%}")
    print(f"  - Ambiguity rate: {mult['ambiguity']['ambiguity_rate']:.2%}")
    print(f"  - Discrepancy: {mult['discrepancy']['discrepancy_empirical']:.2%}")
    print(f"  - Top feature (VIC): {feature_names[top_vic[0]]}")
    print(f"  - Top feature (MCR): {feature_names[top_mcr[0]]}")
    
    if mult['ambiguity']['ambiguity_rate'] > 0.10:
        print("\nWARNING: High ambiguity (>10%) suggests significant model multiplicity.")
        print("  Consider: (1) collecting more data, (2) reducing epsilon, or")
        print("  (3) incorporating domain constraints")
    else:
        print("\nGood news: Low ambiguity suggests predictions are stable across the Rashomon set.")
    
    if mcr['collinearity_warning']:
        print("\nNOTE: Collinearity detected. MCR with 'conditional' permutations is recommended.")
    
    print("\nTotal computation time:")
    total_time = fit_time + vic_time + mcr_time + mult_time
    print(f"  Fit: {fit_time:.2f}s, VIC: {vic_time:.2f}s, MCR: {mcr_time:.2f}s, "
          f"Multiplicity: {mult_time:.2f}s")
    print(f"  Total: {total_time:.2f}s")
    
    print("\n" + "=" * 80)
    print("  Comprehensive demonstration complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()

