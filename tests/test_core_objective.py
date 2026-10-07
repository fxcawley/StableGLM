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
