"""Tests for VIC and MCR metrics (E20, E22)."""
import numpy as np

from rashomon import RashomonSet


def _make_data(n: int = 50, d: int = 5, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    w = rng.normal(size=d)
    logits = X @ w
    p = 1.0 / (1.0 + np.exp(-logits))
    y = (rng.random(n) < p).astype(float)
    return X, y


def test_vic_computation():
    """Test Variable Importance Cloud computation."""
    X, y = _make_data(n=60, d=4, seed=42)
    rs = RashomonSet(
        estimator="logistic",
        epsilon=0.08,
        epsilon_mode="percent_loss",
        random_state=0
    ).fit(X, y)

    # Compute VIC
    vic = rs.variable_importance_cloud(n_samples=50, random_state=123)

    # Check structure
    assert "samples" in vic
    assert "mean" in vic
    assert "std" in vic
    assert "quantiles" in vic
    assert "intervals" in vic
    assert "feature_names" in vic

    # Check shapes
    assert vic["samples"].shape == (50, 4)
    assert vic["mean"].shape == (4,)
    assert vic["std"].shape == (4,)
    assert vic["intervals"].shape == (4, 2)
    assert len(vic["feature_names"]) == 4

    # Check quantiles
    for q in [0.05, 0.25, 0.5, 0.75, 0.95]:
        assert q in vic["quantiles"]
        assert vic["quantiles"][q].shape == (4,)

    # Intervals should be [q_0.05, q_0.95]
    assert np.allclose(vic["intervals"][:, 0], vic["quantiles"][0.05])
    assert np.allclose(vic["intervals"][:, 1], vic["quantiles"][0.95])


def test_vic_with_feature_names():
    """Test VIC with custom feature names."""
    X, y = _make_data(n=50, d=3, seed=99)
    rs = RashomonSet(
        estimator="logistic",
        epsilon=0.05,
        epsilon_mode="percent_loss",
        random_state=0
    ).fit(X, y)

    names = ["age", "income", "score"]
    vic = rs.variable_importance_cloud(
        n_samples=30,
        feature_names=names,
        random_state=456
    )

    assert vic["feature_names"] == names


def test_vic_ellipsoid_vs_hitandrun():
    """Test VIC with both samplers produces similar results."""
    X, y = _make_data(n=50, d=4, seed=77)
    rs = RashomonSet(
        estimator="logistic",
        epsilon=0.1,
        epsilon_mode="percent_loss",
        random_state=0
    ).fit(X, y)

    vic_ell = rs.variable_importance_cloud(
        n_samples=100,
        sampler="ellipsoid",
        random_state=111
    )

    vic_har = rs.variable_importance_cloud(
        n_samples=50,
        sampler="hitandrun",
        burnin=30,
        thin=2,
        random_state=222
    )

    # Both should have similar structure
    assert vic_ell["mean"].shape == vic_har["mean"].shape

    # Means should be reasonably close (within reasonable tolerance due to sampling)
    # This is a weak test - just checking they're in the same ballpark
    mean_diff = np.abs(vic_ell["mean"] - vic_har["mean"])
    assert np.mean(mean_diff) < 1.0  # loose tolerance


def test_vic_legacy_method():
    """Test legacy variable_importance method."""
    X, y = _make_data(n=40, d=3, seed=88)
    rs = RashomonSet(
        estimator="logistic",
        epsilon=0.06,
        epsilon_mode="percent_loss",
        random_state=0
    ).fit(X, y)

    # Legacy method should work
    vic = rs.variable_importance(mode="VIC")
    assert "samples" in vic
    assert "mean" in vic


def test_mcr_enhanced_iid():
    """Test enhanced MCR with iid permutation."""
    X, y = _make_data(n=60, d=4, seed=55)
    rs = RashomonSet(
        estimator="logistic",
        epsilon=0.08,
        epsilon_mode="percent_loss",
        random_state=0
    ).fit(X, y)

    mcr = rs.model_class_reliance(
        X, y,
        n_permutations=8,
        n_samples=20,
        perm_mode="iid",
        random_state=123
    )

    # Check structure
    assert "feature_importance" in mcr
    assert "importance_std" in mcr
    assert "base_score" in mcr
    assert "collinearity_warning" in mcr
    assert "importance_matrix" in mcr

    # Check shapes
    assert mcr["feature_importance"].shape == (4,)
    assert mcr["importance_std"].shape == (4,)
    assert mcr["importance_matrix"].shape == (20, 4)

    # Base score should be reasonable
    assert 0.0 <= mcr["base_score"] <= 1.0


def test_mcr_residual_mode():
    """Test MCR with residual permutation mode."""
    X, y = _make_data(n=50, d=3, seed=66)
    rs = RashomonSet(
        estimator="logistic",
        epsilon=0.1,
        epsilon_mode="percent_loss",
        random_state=0
    ).fit(X, y)

    mcr = rs.model_class_reliance(
        X, y,
        n_permutations=6,
        n_samples=15,
        perm_mode="residual",
        check_collinearity=False,
        random_state=321
    )

    assert mcr["feature_importance"].shape == (3,)
    assert mcr["importance_std"].shape == (3,)


def test_mcr_conditional_mode():
    """Test MCR with conditional permutation mode."""
    X, y = _make_data(n=50, d=3, seed=77)
    rs = RashomonSet(
        estimator="logistic",
        epsilon=0.09,
        epsilon_mode="percent_loss",
        random_state=0
    ).fit(X, y)

    mcr = rs.model_class_reliance(
        X, y,
        n_permutations=5,
        n_samples=10,
        perm_mode="conditional",
        random_state=789
    )

    assert mcr["feature_importance"].shape == (3,)


def test_mcr_collinearity_detection():
    """Test MCR detects collinearity."""
    n, d = 60, 4
    rng = np.random.default_rng(99)
    X = rng.normal(size=(n, d))

    # Make features 0 and 1 highly correlated
    X[:, 1] = 0.95 * X[:, 0] + 0.05 * rng.normal(size=n)

    w = rng.normal(size=d)
    logits = X @ w
    p = 1.0 / (1.0 + np.exp(-logits))
    y = (rng.random(n) < p).astype(float)

    rs = RashomonSet(
        estimator="logistic",
        epsilon=0.1,
        epsilon_mode="percent_loss",
        random_state=0
    ).fit(X, y)

    # Should detect collinearity
    mcr = rs.model_class_reliance(
        X, y,
        n_permutations=4,
        n_samples=10,
        check_collinearity=True,
        random_state=111
    )

    # Should have collinearity warning
    assert mcr["collinearity_warning"] is not None
    assert len(mcr["collinearity_warning"]) > 0


def test_mcr_linear_model():
    """Test MCR works with linear models."""
    n, d = 50, 3
    rng = np.random.default_rng(88)
    X = rng.normal(size=(n, d))
    w = np.array([2.0, -1.5, 0.5])
    y = X @ w + 0.1 * rng.normal(size=n)

    rs = RashomonSet(
        estimator="linear",
        epsilon=0.05,
        epsilon_mode="percent_loss",
        random_state=0
    ).fit(X, y)

    mcr = rs.model_class_reliance(
        X, y,
        n_permutations=5,
        n_samples=15,
        perm_mode="iid",
        random_state=222
    )

    # Feature 0 should be most important (largest coefficient)
    assert mcr["feature_importance"][0] == np.max(mcr["feature_importance"])


def test_vic_plot_runs():
    """Test VIC plotting doesn't crash (visual check needed separately)."""
    X, y = _make_data(n=50, d=3, seed=11)
    rs = RashomonSet(
        estimator="logistic",
        epsilon=0.1,
        epsilon_mode="percent_loss",
        random_state=0
    ).fit(X, y)

    try:
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend

        fig, ax = rs.plot_vic(n_samples=30, random_state=123)
        assert fig is not None
        assert ax is not None

        # Can also pass pre-computed VIC
        vic = rs.variable_importance_cloud(n_samples=30, random_state=123)
        fig2, ax2 = rs.plot_vic(vic_result=vic)
        assert fig2 is not None
        assert ax2 is not None

    except ImportError:
        # Matplotlib not available - skip
        pass

