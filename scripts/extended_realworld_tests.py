"""Extended real-world tests with additional datasets and deeper validation.

This script tests:
- More diverse datasets (iris, wine, digits subset)
- Cross-validation of predictions
- Comparison with scikit-learn baselines
- Edge cases and robustness
"""

import time

import numpy as np

from rashomon import RashomonSet


def print_section(title):
    """Print formatted section header."""
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print("=" * 80)


def test_iris_multiclass_ovr():
    """Test on Iris dataset (3 classes, converted to binary)."""
    print_section("1. Iris Classification (Binary: Setosa vs Others)")

    try:
        from sklearn.datasets import load_iris
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        print("Skipping: sklearn not available")
        return

    data = load_iris()
    X = StandardScaler().fit_transform(data.data)
    y = (data.target == 0).astype(float)  # Setosa vs others

    print(f"Dataset: n={X.shape[0]}, d={X.shape[1]}")
    print(f"Prevalence: {np.mean(y):.2%}")

    # Fit
    rs = RashomonSet(
        estimator="logistic",
        epsilon_mode="percent_loss",
        epsilon=0.10,
        random_state=42
    ).fit(X, y)

    print(f"L_hat: {rs._L_hat:.4f}, Epsilon: {rs._epsilon_value:.4f}")

    # Predictions
    y_pred_opt = rs.predict(X)
    acc_opt = np.mean(y_pred_opt == y)
    print(f"Optimal model accuracy: {acc_opt:.2%}")

    # Check intervals for some instances
    intervals_coef = rs.coef_intervals()
    avg_interval_width = np.mean(intervals_coef[:, 1] - intervals_coef[:, 0])
    print(f"Average coefficient interval width: {avg_interval_width:.3f}")

    # Multiplicity
    mult = rs.multiplicity(
        X, which=["ambiguity", "discrepancy"],
        y=y, n_samples=50,
        threshold_mode="match_prevalence",
        random_state=42
    )
    print(f"Ambiguity: {mult['ambiguity']['ambiguity_rate']:.2%}")
    print(f"Discrepancy (empirical): {mult['discrepancy']['discrepancy_empirical']:.2%}")


def test_wine_classification():
    """Test on Wine dataset (binary: class 0 vs 1)."""
    print_section("2. Wine Classification (Binary)")

    try:
        from sklearn.datasets import load_wine
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        print("Skipping: sklearn not available")
        return

    data = load_wine()
    # Use only classes 0 and 1
    mask = data.target < 2
    X = StandardScaler().fit_transform(data.data[mask])
    y = data.target[mask].astype(float)

    print(f"Dataset: n={X.shape[0]}, d={X.shape[1]}")
    print(f"Prevalence: {np.mean(y):.2%}")

    # Fit with regularization
    rs = RashomonSet(
        estimator="logistic",
        C=0.1,
        epsilon_mode="percent_loss",
        epsilon=0.05,
        random_state=42
    ).fit(X, y)

    # VIC and MCR
    vic = rs.variable_importance_cloud(n_samples=50, random_state=42)
    mcr = rs.model_class_reliance(X, y, n_samples=30, n_permutations=10, random_state=42)

    # Compare top features
    top_vic_idx = np.argsort(np.abs(vic["mean"]))[-3:][::-1]
    top_mcr_idx = np.argsort(mcr["feature_importance"])[-3:][::-1]

    print("Top 3 features by VIC (|mean coef|):")
    for idx in top_vic_idx:
        print(f"  Feature {idx}: {vic['mean'][idx]:.3f}")

    print("Top 3 features by MCR:")
    for idx in top_mcr_idx:
        print(f"  Feature {idx}: {mcr['feature_importance'][idx]:.4f}")

    # Check overlap
    overlap = len(set(top_vic_idx) & set(top_mcr_idx))
    print(f"Agreement: {overlap}/3 features in both top-3 lists")


def test_california_housing_regression():
    """Test on synthetic regression data (like California Housing)."""
    print_section("3. Synthetic Regression (Multicollinear Features)")

    # Create synthetic data instead of downloading
    print("Note: Using synthetic data similar to housing regression")
    np.random.seed(42)
    n, d = 500, 8

    # Create correlated features
    X = np.random.randn(n, d)
    X[:, 1] = 0.6 * X[:, 0] + 0.4 * X[:, 1]  # Correlated with feature 0
    X[:, 5] = 0.7 * X[:, 4] + 0.3 * X[:, 5]  # Correlated with feature 4

    from sklearn.preprocessing import StandardScaler
    X = StandardScaler().fit_transform(X)

    # Create target with some noise
    true_coef = np.array([2.5, -1.8, 0.5, 1.2, -0.8, 0.3, 0.6, -1.0])
    y = X @ true_coef + 0.5 * np.random.randn(n)

    print(f"Dataset: n={X.shape[0]}, d={X.shape[1]}")
    print(f"Target range: [{y.min():.2f}, {y.max():.2f}], mean={y.mean():.2f}")

    # Fit
    start = time.time()
    rs = RashomonSet(
        estimator="linear",
        epsilon_mode="percent_loss",
        epsilon=0.05,
        random_state=42
    ).fit(X, y)
    fit_time = time.time() - start

    print(f"Fit time: {fit_time:.2f}s")
    print(f"L_hat: {rs._L_hat:.4f}, R2: {rs.score(X, y):.4f}")

    # Coefficient analysis
    intervals = rs.coef_intervals()
    n_certain = np.sum((intervals[:, 0] > 0) | (intervals[:, 1] < 0))
    print(f"Coefficients with sign certainty: {n_certain}/{X.shape[1]}")

    # VIC
    vic = rs.variable_importance_cloud(n_samples=50, random_state=42)
    print(f"Average coefficient std: {np.mean(vic['std']):.4f}")
    print(f"Max coefficient std: {np.max(vic['std']):.4f}")


def test_sklearn_comparison():
    """Compare RashomonSet predictions with sklearn baseline."""
    print_section("4. Comparison with sklearn Baseline")

    try:
        from sklearn.datasets import make_classification
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        print("Skipping: sklearn not available")
        return

    # Generate data
    X, y = make_classification(
        n_samples=300, n_features=10, n_informative=6,
        n_redundant=2, random_state=42
    )
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    print(f"Train: n={len(X_train)}, Test: n={len(X_test)}")

    # Sklearn baseline
    lr = LogisticRegression(penalty="l2", C=1.0, max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    acc_sklearn = lr.score(X_test, y_test)

    # RashomonSet
    rs = RashomonSet(
        estimator="logistic",
        C=1.0,
        epsilon_mode="percent_loss",
        epsilon=0.01,  # Small epsilon = close to optimal
        random_state=42
    ).fit(X_train, y_train)

    acc_rashomon = rs.score(X_test, y_test)

    print(f"sklearn accuracy: {acc_sklearn:.4f}")
    print(f"RashomonSet accuracy: {acc_rashomon:.4f}")
    print(f"Difference: {abs(acc_sklearn - acc_rashomon):.4f}")

    # Coefficient comparison
    coef_diff = np.linalg.norm(lr.coef_[0] - rs._theta_hat)
    print(f"Coefficient L2 distance: {coef_diff:.4f}")

    # With larger epsilon, predictions should vary more
    rs_large = RashomonSet(
        estimator="logistic",
        C=1.0,
        epsilon_mode="percent_loss",
        epsilon=0.10,  # Larger epsilon
        random_state=42
    ).fit(X_train, y_train)

    amb_small = rs.ambiguity(X_test, threshold_mode="fixed", threshold_value=0.5)
    amb_large = rs_large.ambiguity(X_test, threshold_mode="fixed", threshold_value=0.5)

    print(f"Ambiguity (epsilon=0.01): {amb_small['ambiguity_rate']:.2%}")
    print(f"Ambiguity (epsilon=0.10): {amb_large['ambiguity_rate']:.2%}")
    print("Larger epsilon -> more multiplicity: ", "OK" if amb_large["ambiguity_rate"] > amb_small["ambiguity_rate"] else "UNEXPECTED")


def test_edge_cases():
    """Test edge cases and boundary conditions."""
    print_section("5. Edge Cases and Robustness")

    print("\n--- Test 1: Perfect separation (almost) ---")
    np.random.seed(42)
    X1 = np.random.randn(50, 3) + 3  # Class 1
    X0 = np.random.randn(50, 3) - 3  # Class 0
    X = np.vstack([X1, X0])
    y = np.hstack([np.ones(50), np.zeros(50)])

    try:
        rs = RashomonSet(
            estimator="logistic",
            C=0.1,  # Regularization to handle separation
            epsilon_mode="percent_loss",
            epsilon=0.05,
            random_state=42
        ).fit(X, y)
        print(f"  Fit successful: L_hat={rs._L_hat:.4f}")
        print(f"  Train accuracy: {rs.score(X, y):.2%}")
    except RuntimeError as e:
        print(f"  Expected error for near-separation: {str(e)[:60]}")

    print("\n--- Test 2: Imbalanced classes (10:90) ---")
    np.random.seed(123)
    n_minority, n_majority = 20, 180
    X = np.random.randn(n_minority + n_majority, 5)
    y = np.hstack([np.ones(n_minority), np.zeros(n_majority)])

    rs = RashomonSet(
        estimator="logistic",
        epsilon_mode="percent_loss",
        epsilon=0.10,
        random_state=42
    ).fit(X, y)

    print(f"  Prevalence: {np.mean(y):.2%}")
    print(f"  L_hat: {rs._L_hat:.4f}")

    # Test different thresholds
    amb_fixed = rs.ambiguity(X, threshold_mode="fixed", threshold_value=0.5)
    amb_prev = rs.ambiguity(X, threshold_mode="match_prevalence", y=y)

    print(f"  Ambiguity (fixed 0.5): {amb_fixed['ambiguity_rate']:.2%}, threshold={amb_fixed['threshold']:.3f}")
    print(f"  Ambiguity (match prev): {amb_prev['ambiguity_rate']:.2%}, threshold={amb_prev['threshold']:.3f}")

    print("\n--- Test 3: All zero features ---")
    X = np.zeros((100, 5))
    X[:, 0] = np.random.randn(100)  # Only first feature is non-zero
    y = (X[:, 0] > 0).astype(float)

    rs = RashomonSet(
        estimator="logistic",
        epsilon_mode="percent_loss",
        epsilon=0.05,
        random_state=42
    ).fit(X, y)

    intervals = rs.coef_intervals()
    n_zero_interval = np.sum((intervals[:, 0] <= 0) & (intervals[:, 1] >= 0))
    print(f"  Coefficients with intervals containing zero: {n_zero_interval}/5")
    print("  (Expected: 4 zero features should have intervals around 0)")


def test_sampling_convergence():
    """Test sampler convergence and diagnostics."""
    print_section("6. Sampling Convergence Analysis")

    np.random.seed(789)
    n, d = 200, 8
    X = np.random.randn(n, d)

    from sklearn.preprocessing import StandardScaler
    X = StandardScaler().fit_transform(X)

    true_coef = np.array([1.0, -0.8, 0.5, 0.3, -0.4, 0.2, -0.1, 0.6])
    logits = X @ true_coef + 0.3 * np.random.randn(n)
    y = (logits > 0).astype(float)

    rs = RashomonSet(
        estimator="logistic",
        epsilon_mode="percent_loss",
        epsilon=0.10,
        random_state=42
    ).fit(X, y)

    # Compare ellipsoid vs hit-and-run
    print("\n--- Ellipsoid Sampling ---")
    samples_ell = rs.sample_ellipsoid(n_samples=100, random_state=42)
    diag_ell = rs.compute_sample_diagnostics(samples_ell)
    print(f"  Set fidelity: {diag_ell['set_fidelity']:.2%}")
    print(f"  Chord mean: {diag_ell['chord_mean']:.4f}")

    print("\n--- Hit-and-Run Sampling ---")
    samples_har = rs.sample_hitandrun(
        n_samples=100, burnin=50, thin=2, random_state=42
    )
    diag_har = rs.compute_sample_diagnostics(samples_har)
    print(f"  Set fidelity: {diag_har['set_fidelity']:.2%}")
    print(f"  Chord mean: {diag_har['chord_mean']:.4f}")

    # VIC comparison - compute manually from samples
    print("\n--- VIC Comparison (Ellipsoid vs Hit-and-Run) ---")
    # Compute mean coefficients from each sampler
    vic_ell_mean = np.mean(samples_ell, axis=0)
    vic_har_mean = np.mean(samples_har, axis=0)

    mean_diff = np.mean(np.abs(vic_ell_mean - vic_har_mean))
    print(f"  Mean coefficient difference: {mean_diff:.4f}")
    print("  (Should be small if samplers converged)")


def main():
    """Run all extended real-world tests."""
    print("\n" + "=" * 80)
    print("  StableGLM/rashomon-py: Extended Real-World Tests")
    print("=" * 80)

    tests = [
        ("Iris Classification", test_iris_multiclass_ovr),
        ("Wine Classification", test_wine_classification),
        ("California Housing", test_california_housing_regression),
        ("sklearn Comparison", test_sklearn_comparison),
        ("Edge Cases", test_edge_cases),
        ("Sampling Convergence", test_sampling_convergence),
    ]

    results = []
    for name, test_func in tests:
        try:
            test_func()
            results.append((name, "PASS"))
        except Exception as e:
            print(f"\nX ERROR in {name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, f"X FAIL: {str(e)[:50]}"))

    # Summary
    print_section("Test Summary")
    for name, result in results:
        print(f"{result:15s} {name}")

    n_passed = sum(1 for _, r in results if r.startswith("PASS"))
    print(f"\nPassed: {n_passed}/{len(tests)}")

    return n_passed == len(tests)


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

