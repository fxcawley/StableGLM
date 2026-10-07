"""Core-objective tests: penalty mask, lambda=0, solver accuracy, samplers, exact ranges."""

import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

from rashomon import RashomonSet


@pytest.fixture(scope="module")
def cancer():
    X, y = load_breast_cancer(return_X_y=True)
    return StandardScaler().fit_transform(X[:, :10]), y.astype(float)


def test_ellipsoid_samples_lie_in_the_hessian_ellipsoid(cancer):
    """Regression test: v0.1 solved with L instead of L^T and sampled a mis-oriented ellipsoid."""
    X, y = cancer
    rs = RashomonSet(estimator="logistic", C=5.0, epsilon=0.03, random_state=0).fit(X, y)
    S = rs.sample_ellipsoid(3000, random_state=0)
    D = S - rs._theta_hat
    quad = np.einsum("ij,jk,ik->i", D, rs._hessian_matrix(), D)
    assert np.all(quad <= 2.0 * rs._epsilon_value * (1 + 1e-9))
    # the ellipsoid approximates the true set well at this dimension
    assert rs.contains_many(S).mean() > 0.8
    # uniform in the ellipsoid: radius^d is Uniform(0,1) -> mean of (quad/2eps)^(d/2) is 1/2
    d = rs._d
    assert np.mean((quad / (2.0 * rs._epsilon_value)) ** (d / 2)) == pytest.approx(0.5, abs=0.03)


def test_hybrid_sampler_matches_pure_hit_and_run_and_exact_linear_case(cancer):
    X, y01 = cancer
    n = len(y01)
    # Logistic: hybrid and pure chains target the same law
    rs = RashomonSet(estimator="logistic", C=5.0, epsilon=0.03, random_state=0).fit(X, y01)
    pure = rs.sample_hitandrun(n_samples=1500, burnin=100, random_state=1)
    ess_pure = rs._last_sample_diagnostics["ess_per_param"].min()
    hybrid = rs.sample_hitandrun(n_samples=1500, burnin=100, random_state=2, ellipsoid_mix=0.5)
    diag = rs._last_sample_diagnostics
    assert diag["ellipsoid_acceptance"] > 0.5
    assert diag["ess_per_param"].min() > ess_pure
    assert rs.contains_many(hybrid).all()
    sd = pure.std(axis=0)
    assert (np.abs(pure.mean(axis=0) - hybrid.mean(axis=0)) / sd).max() < 0.35
    assert np.abs(hybrid.std(axis=0) / sd - 1).max() < 0.25
    # Linear: the set *is* the ellipsoid, so every proposal is accepted
    rng = np.random.default_rng(0)
    y = X @ np.arange(1, 11) + 3 * rng.normal(size=n) + 5.0
    rl = RashomonSet(estimator="linear", C=n / 3.0, fit_intercept=True, epsilon=0.05, random_state=0).fit(X, y)
    rl.sample_hitandrun(n_samples=300, burnin=50, random_state=3, ellipsoid_mix=0.5)
    assert rl._last_sample_diagnostics["ellipsoid_acceptance"] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="ellipsoid_mix"):
        rl.sample_hitandrun(n_samples=10, ellipsoid_mix=1.0)
