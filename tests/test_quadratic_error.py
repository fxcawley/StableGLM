"""quadratic_error: how far the exact Rashomon range departs from the Hessian ellipsoid, estimated
along the ellipsoid's extremal rays (exact line search on the loss, or its quartic Taylor model)."""

import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from rashomon import RashomonSet


@pytest.fixture(scope="module")
def cancer():
    data = load_breast_cancer()
    X = StandardScaler().fit_transform(data.data[:, :10])
    y = data.target.astype(float)
    return LogisticRegression().fit(X, y), X, y


def _fit(cancer, **kw):
    model, X, y = cancer
    return RashomonSet.from_sklearn(model, X, y, random_state=0, **kw)


def _observed_deviation(rs):
    r = np.sqrt(2 * rs._epsilon_value)
    half = r * np.sqrt(np.diag(np.linalg.inv(rs._hessian_matrix())))
    ex, th = rs.coef_extremes(), rs._theta_hat
    return np.c_[(th - ex[:, 0]) / half - 1, (ex[:, 1] - th) / half - 1]


def test_linear_model_has_no_error():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 4))
    y = X @ np.array([1.0, -0.5, 0.0, 0.3]) + rng.normal(size=200)
    rs = RashomonSet(estimator="linear", C=10.0, epsilon=0.05).fit(X, y)
    assert np.all(rs.quadratic_error() == 0.0)
    assert np.allclose(rs.coef_intervals(), rs.coef_extremes(), rtol=1e-6)


def test_ray_boundary_lies_on_the_level_set(cancer):
    """The ray method's boundary point has loss gap exactly eps; the quartic model's is close;
    the ellipsoid's is not."""
    rs = _fit(cancer, epsilon=0.1, epsilon_mode="percent_loss")
    eps, r = rs._epsilon_value, np.sqrt(2 * rs._epsilon_value)
    Hinv = np.linalg.inv(rs._hessian_matrix())
    dev = {m: rs.quadratic_error(method=m) for m in ("ray", "quartic")}
    miss = {"ray": [], "quartic": [], "ellipsoid": []}
    for j in range(rs._d):
        u = Hinv[:, j] / np.sqrt(Hinv[j, j])
        for side, k in ((-1, 0), (1, 1)):
            for m in ("ray", "quartic"):
                t = side * r * (1 + dev[m][j, k])
                miss[m].append(abs(rs.objective(rs._theta_hat + t * u) - rs._L_hat - eps))
            miss["ellipsoid"].append(abs(rs.objective(rs._theta_hat + side * r * u) - rs._L_hat - eps))
    assert max(miss["ray"]) < 1e-9 * eps
    assert max(miss["quartic"]) < 0.05 * eps
    assert max(miss["quartic"]) < 0.2 * max(miss["ellipsoid"])


@pytest.mark.parametrize("eps_over_loss, ellipsoid_error, residual", [(0.02, 0.04, 0.01), (0.12, 0.10, 0.05)])
def test_ray_predicts_exact_coefficient_ranges(cancer, eps_over_loss, ellipsoid_error, residual):
    """The ellipsoid is off by up to `ellipsoid_error`; the ray estimate is within `residual` of
    the exact ranges and never exceeds them (its boundary points are in the set)."""
    rs = _fit(cancer, epsilon=eps_over_loss, epsilon_mode="percent_loss")
    observed = _observed_deviation(rs)
    ray = rs.quadratic_error()
    assert np.abs(observed).max() > ellipsoid_error
    assert np.all(observed - ray > -1e-6)
    assert np.abs(observed - ray).max() < residual
    quartic = rs.quadratic_error(method="quartic")
    assert np.abs(observed - quartic).max() < np.abs(observed).max()


def test_newton_step_from_the_ray_nearly_closes_the_gap(cancer):
    """One KKT step from the ray boundary point: still inside the set, within 1% of the exact
    ranges where the ray alone leaves 4%; two steps within 0.2%."""
    rs = _fit(cancer, epsilon=0.12, epsilon_mode="percent_loss")
    observed = _observed_deviation(rs)
    ray = rs.quadratic_error()
    one = rs.quadratic_error(method="newton")
    two = rs.quadratic_error(method="newton", steps=2)
    for est in (one, two):
        assert np.all(est >= ray - 1e-9)  # never worse than the ray
        assert np.all(observed - est > -1e-5)  # still an inner bound
    assert np.abs(observed - ray).max() > 0.03
    assert np.abs(observed - one).max() < 0.01
    assert np.abs(observed - two).max() < 0.002


def test_functionals_and_shapes(cancer):
    rs = _fit(cancer, epsilon=0.02)
    X = rs._prepare_X(cancer[1])
    dev = rs.quadratic_error(X[:7])
    assert dev.shape == (7, 2) and np.all(np.isfinite(dev))
    assert rs.quadratic_error(X[0]).shape == (1, 2)
    # a row's exact logit range is at least the ray estimate on both sides
    r = np.sqrt(2 * rs._epsilon_value)
    for i in range(3):
        norm = np.sqrt(X[i] @ np.linalg.solve(rs._hessian_matrix(), X[i]))
        lo, hi = rs.functional_range(X[i])
        m = X[i] @ rs._theta_hat
        assert m - lo >= r * norm * (1 + dev[i, 0]) - 1e-8
        assert hi - m >= r * norm * (1 + dev[i, 1]) - 1e-8
    # the newton estimate for rows is also inside the exact logit range
    dev_n = rs.quadratic_error(X[:3], method="newton")
    for i in range(3):
        norm = np.sqrt(X[i] @ np.linalg.solve(rs._hessian_matrix(), X[i]))
        lo, hi = rs.functional_range(X[i])
        m = X[i] @ rs._theta_hat
        assert m - lo >= r * norm * (1 + dev_n[i, 0]) - 1e-6
        assert hi - m >= r * norm * (1 + dev_n[i, 1]) - 1e-6
    with pytest.raises(ValueError):
        rs.quadratic_error(np.ones((2, rs._d + 1)))
    with pytest.raises(ValueError):
        rs.quadratic_error(method="cubic")
