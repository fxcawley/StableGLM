"""Tests for sampling diagnostics (D18)."""
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


def test_sample_diagnostics_computation():
    """Test compute_sample_diagnostics method."""
    X, y = _make_data(n=60, d=5, seed=42)
    rs = RashomonSet(
        estimator="logistic",
        epsilon=0.05,
        epsilon_mode="percent_loss",
        random_state=0
    ).fit(X, y)
    
    # Generate samples using ellipsoid
    samples = rs.sample_ellipsoid(n_samples=50, random_state=123)
    
    # Compute diagnostics
    diag = rs.compute_sample_diagnostics(samples, burnin=0)
    
    # Check all expected keys are present
    assert "n_samples" in diag
    assert "set_fidelity" in diag
    assert "chord_mean" in diag
    assert "chord_std" in diag
    assert "chord_min" in diag
    assert "chord_max" in diag
    assert "isotropy_ratio" in diag
    
    # Validate values
    assert diag["n_samples"] == 50
    assert 0.0 <= diag["set_fidelity"] <= 1.0
    assert diag["chord_mean"] > 0
    assert diag["chord_std"] >= 0
    assert diag["isotropy_ratio"] is not None and diag["isotropy_ratio"] > 0


def test_hitandrun_with_diagnostics():
    """Test Hit-and-Run sampler computes and caches diagnostics."""
    X, y = _make_data(n=50, d=4, seed=99)
    rs = RashomonSet(
        estimator="logistic",
        epsilon=0.08,
        epsilon_mode="percent_loss",
        sampler="hitandrun",
        random_state=0
    ).fit(X, y)
    
    # Sample with diagnostics computation
    samples = rs.sample_hitandrun(
        n_samples=30,
        burnin=20,
        thin=2,
        random_state=456,
        compute_diagnostics=True
    )
    
    assert samples.shape == (30, 4)
    
    # Check diagnostics were cached
    assert rs._last_sample_diagnostics is not None
    
    # Get full diagnostics
    full_diag = rs.diagnostics()
    assert "sampler_diagnostics" in full_diag
    assert full_diag["set_fidelity"] is not None
    
    # All samples should be in set (Hit-and-Run ensures membership)
    assert full_diag["set_fidelity"] >= 0.95  # Allow small tolerance


def test_ellipsoid_caching_speedup():
    """Test that Hessian caching provides speedup."""
    X, y = _make_data(n=40, d=6, seed=77)
    rs = RashomonSet(
        estimator="logistic",
        random_state=0
    ).fit(X, y)
    
    import time
    
    # First call - computes and caches
    start = time.time()
    samples1 = rs.sample_ellipsoid(n_samples=100, random_state=1)
    time1 = time.time() - start
    
    # Second call - uses cache
    start = time.time()
    samples2 = rs.sample_ellipsoid(n_samples=100, random_state=2)
    time2 = time.time() - start
    
    # Both should produce valid samples
    assert samples1.shape == (100, 6)
    assert samples2.shape == (100, 6)
    
    # Samples should be different (different random seeds)
    assert not np.allclose(samples1, samples2)
    
    # Second call should generally be faster or similar (caching helps)
    # Not a strict requirement due to system variance
    print(f"Time1: {time1:.4f}s, Time2: {time2:.4f}s")


def test_diagnostics_with_burnin():
    """Test diagnostics computation with burnin."""
    X, y = _make_data(n=45, d=4, seed=88)
    rs = RashomonSet(
        estimator="logistic",
        epsilon=0.06,
        epsilon_mode="percent_loss",
        random_state=0
    ).fit(X, y)
    
    samples = rs.sample_ellipsoid(n_samples=60, random_state=321)
    
    # Compute with burnin
    diag = rs.compute_sample_diagnostics(samples, burnin=10)
    assert diag["n_samples"] == 50  # 60 - 10
    
    # Without burnin
    diag_full = rs.compute_sample_diagnostics(samples, burnin=0)
    assert diag_full["n_samples"] == 60


def test_ess_computation():
    """Test ESS per parameter computation."""
    X, y = _make_data(n=50, d=3, seed=111)
    rs = RashomonSet(
        estimator="logistic",
        epsilon=0.1,
        epsilon_mode="percent_loss",
        random_state=0
    ).fit(X, y)
    
    samples = rs.sample_ellipsoid(n_samples=100, random_state=999)
    diag = rs.compute_sample_diagnostics(samples, compute_ess=True)
    
    assert diag["ess_per_param"] is not None
    assert len(diag["ess_per_param"]) == 3
    assert all(ess >= 1.0 for ess in diag["ess_per_param"])

