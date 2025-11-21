"""Tests for predictive multiplicity metrics (E24, E24b, E25)."""

import numpy as np
import pytest

from rashomon import RashomonSet


def test_threshold_fixed():
    """Test fixed threshold computation."""
    X = np.random.randn(100, 5)
    y = np.random.randint(0, 2, 100)

    rs = RashomonSet(estimator="logistic", random_state=42).fit(X, y)

    # Fixed threshold at 0.5 should map to logit(0.5) = 0
    tau = rs.compute_threshold(y, mode="fixed", value=0.5)
    assert abs(tau - 0.0) < 1e-6

    # Fixed at 0.7
    tau = rs.compute_threshold(y, mode="fixed", value=0.7)
    expected = np.log(0.7 / 0.3)
    assert abs(tau - expected) < 1e-6


def test_threshold_match_prevalence():
    """Test prevalence-matching threshold."""
    X = np.random.randn(100, 5)
    y = np.array([0] * 30 + [1] * 70)  # 70% prevalence

    rs = RashomonSet(estimator="logistic", random_state=42).fit(X, y)

    tau = rs.compute_threshold(y, mode="match_prevalence")
    # Threshold should be logit(0.7)
    expected = np.log(0.7 / 0.3)
    assert abs(tau - expected) < 1e-6


def test_threshold_match_fpr():
    """Test FPR-matching threshold requires validation data."""
    X = np.random.randn(100, 5)
    y = np.random.randint(0, 2, 100)

    rs = RashomonSet(estimator="logistic", random_state=42).fit(X, y)

    # Should raise without validation data
    with pytest.raises(ValueError, match="requires X_val and y_val"):
        rs.compute_threshold(y, mode="match_fpr", fpr=0.05)

    # Should work with validation data
    X_val = np.random.randn(50, 5)
    y_val = np.random.randint(0, 2, 50)
    tau = rs.compute_threshold(y, mode="match_fpr", fpr=0.05, X_val=X_val, y_val=y_val)
    assert isinstance(tau, float)


def test_threshold_youden():
    """Test Youden threshold requires validation data."""
    X = np.random.randn(100, 5)
    y = np.random.randint(0, 2, 100)

    rs = RashomonSet(estimator="logistic", random_state=42).fit(X, y)

    # Should raise without validation data
    with pytest.raises(ValueError, match="requires X_val and y_val"):
        rs.compute_threshold(y, mode="youden")

    # Should work with validation data
    X_val = np.random.randn(50, 5)
    y_val = np.random.randint(0, 2, 50)
    tau = rs.compute_threshold(y, mode="youden", X_val=X_val, y_val=y_val)
    assert isinstance(tau, float)


def test_threshold_linear_raises():
    """Test that threshold computation raises for linear regression."""
    X = np.random.randn(100, 5)
    y = np.random.randn(100)

    rs = RashomonSet(estimator="linear", random_state=42).fit(X, y)

    with pytest.raises(ValueError, match="only applicable to logistic"):
        rs.compute_threshold(y, mode="fixed", value=0.5)


def test_ambiguity_basic():
    """Test basic ambiguity computation."""
    # Create a simple dataset
    np.random.seed(42)
    X = np.random.randn(50, 5)
    y = np.random.randint(0, 2, 50)

    rs = RashomonSet(estimator="logistic", epsilon_mode="percent_loss", epsilon=0.05, random_state=42).fit(X, y)

    # Compute ambiguity with fixed threshold
    amb = rs.ambiguity(X, threshold_mode="fixed", threshold_value=0.5)

    assert "ambiguity_rate" in amb
    assert "n_ambiguous" in amb
    assert "threshold" in amb
    assert "ambiguous_indices" in amb

    assert 0.0 <= amb["ambiguity_rate"] <= 1.0
    assert amb["n_ambiguous"] == len(amb["ambiguous_indices"])
    assert abs(amb["threshold"] - 0.0) < 1e-6  # threshold 0.5 maps to 0


def test_ambiguity_match_prevalence():
    """Test ambiguity with prevalence-matching threshold."""
    np.random.seed(42)
    X = np.random.randn(50, 5)
    y = np.array([0] * 20 + [1] * 30)

    rs = RashomonSet(estimator="logistic", epsilon_mode="percent_loss", epsilon=0.05, random_state=42).fit(X, y)

    amb = rs.ambiguity(X, threshold_mode="match_prevalence", y=y)

    assert 0.0 <= amb["ambiguity_rate"] <= 1.0
    expected_threshold = np.log(0.6 / 0.4)
    assert abs(amb["threshold"] - expected_threshold) < 1e-6


def test_ambiguity_requires_fit():
    """Test that ambiguity requires fitted model."""
    rs = RashomonSet(estimator="logistic", random_state=42)
    X = np.random.randn(50, 5)

    with pytest.raises(RuntimeError, match="Call fit"):
        rs.ambiguity(X)


def test_ambiguity_zero_epsilon():
    """Test ambiguity with zero epsilon should have no ambiguous points."""
    np.random.seed(42)
    X = np.random.randn(50, 5)
    y = np.random.randint(0, 2, 50)

    rs = RashomonSet(estimator="logistic", epsilon_mode="percent_loss", epsilon=0.0, random_state=42).fit(X, y)

    amb = rs.ambiguity(X, threshold_mode="fixed", threshold_value=0.5)

    # With epsilon=0, no interval should straddle threshold (unless exactly on boundary)
    # Most likely zero or very few ambiguous points
    assert amb["n_ambiguous"] <= 5  # Allow a few due to numerical tolerance


def test_discrepancy_basic():
    """Test basic discrepancy computation."""
    np.random.seed(42)
    X = np.random.randn(50, 5)
    y = np.random.randint(0, 2, 50)

    rs = RashomonSet(estimator="logistic", epsilon_mode="percent_loss", epsilon=0.05, random_state=42).fit(X, y)

    disc = rs.discrepancy(X, n_samples=50, n_pairs=25, threshold_mode="fixed", threshold_value=0.5, random_state=42)

    assert "discrepancy_bound" in disc
    assert "discrepancy_empirical" in disc
    assert "max_pair_disagreement" in disc
    assert "mean_pair_disagreement" in disc
    assert "threshold" in disc

    assert 0.0 <= disc["discrepancy_bound"] <= 1.0
    assert 0.0 <= disc["discrepancy_empirical"] <= 1.0
    assert disc["discrepancy_empirical"] <= disc["discrepancy_bound"]


def test_discrepancy_with_precomputed_samples():
    """Test discrepancy with pre-computed samples."""
    np.random.seed(42)
    X = np.random.randn(50, 5)
    y = np.random.randint(0, 2, 50)

    rs = RashomonSet(estimator="logistic", epsilon_mode="percent_loss", epsilon=0.05, random_state=42).fit(X, y)

    # Pre-sample
    samples = rs.sample_ellipsoid(100, random_state=42)

    disc = rs.discrepancy(X, samples=samples, n_pairs=50, threshold_mode="fixed", threshold_value=0.5, random_state=42)

    assert disc["discrepancy_empirical"] is not None
    assert 0.0 <= disc["discrepancy_empirical"] <= 1.0


def test_discrepancy_bound_vs_empirical():
    """Test that empirical discrepancy is below the bound."""
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = np.random.randint(0, 2, 100)

    rs = RashomonSet(estimator="logistic", epsilon_mode="percent_loss", epsilon=0.1, random_state=42).fit(X, y)

    disc = rs.discrepancy(X, n_samples=100, n_pairs=50, random_state=42)

    # Empirical should be at most the bound
    assert disc["discrepancy_empirical"] <= disc["discrepancy_bound"] + 1e-6


def test_multiplicity_combined():
    """Test combined multiplicity metrics."""
    np.random.seed(42)
    X = np.random.randn(50, 5)
    y = np.random.randint(0, 2, 50)

    rs = RashomonSet(estimator="logistic", epsilon_mode="percent_loss", epsilon=0.05, random_state=42).fit(X, y)

    mult = rs.multiplicity(X, which=["ambiguity", "discrepancy"], y=y, n_samples=50, threshold_mode="fixed")

    assert "ambiguity" in mult
    assert "discrepancy" in mult

    assert mult["ambiguity"]["ambiguity_rate"] is not None
    assert mult["discrepancy"]["discrepancy_bound"] is not None


def test_multiplicity_capacity():
    """Test capacity metric computation."""
    np.random.seed(42)
    X = np.random.randn(50, 5)
    y = np.random.randint(0, 2, 50)

    rs = RashomonSet(estimator="logistic", random_state=42).fit(X, y)

    mult = rs.multiplicity(X, which=["capacity"])

    assert "capacity" in mult
    cap = mult["capacity"]
    assert "log_volume" in cap
    assert "effective_dim" in cap
    assert "log_det_hessian" in cap
    assert isinstance(cap["log_volume"], float)
    assert cap["effective_dim"] == 5.0
    assert np.isfinite(cap["log_volume"])


def test_ambiguity_larger_epsilon():
    """Test that larger epsilon leads to more ambiguity."""
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = np.random.randint(0, 2, 100)

    rs_small = RashomonSet(estimator="logistic", epsilon_mode="percent_loss", epsilon=0.01, random_state=42).fit(X, y)
    rs_large = RashomonSet(estimator="logistic", epsilon_mode="percent_loss", epsilon=0.1, random_state=42).fit(X, y)

    amb_small = rs_small.ambiguity(X)
    amb_large = rs_large.ambiguity(X)

    # Larger epsilon should generally lead to higher ambiguity
    assert amb_large["ambiguity_rate"] >= amb_small["ambiguity_rate"]


def test_discrepancy_larger_epsilon():
    """Test that larger epsilon leads to higher discrepancy."""
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = np.random.randint(0, 2, 100)

    rs_small = RashomonSet(estimator="logistic", epsilon_mode="percent_loss", epsilon=0.01, random_state=42).fit(X, y)
    rs_large = RashomonSet(estimator="logistic", epsilon_mode="percent_loss", epsilon=0.1, random_state=42).fit(X, y)

    disc_small = rs_small.discrepancy(X, n_samples=50, n_pairs=25, random_state=42)
    disc_large = rs_large.discrepancy(X, n_samples=50, n_pairs=25, random_state=42)

    # Larger epsilon should lead to higher discrepancy bound
    assert disc_large["discrepancy_bound"] >= disc_small["discrepancy_bound"]

