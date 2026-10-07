"""Core-objective tests: penalty mask, lambda=0, solver accuracy, samplers, exact ranges."""

import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

from rashomon import RashomonSet


@pytest.fixture(scope="module")
def cancer():
    X, y = load_breast_cancer(return_X_y=True)
    return StandardScaler().fit_transform(X[:, :10]), y.astype(float)


def test_unpenalized_intercept_matches_sklearn(cancer):
    X, y = cancer
    n = len(y)
    sk = LogisticRegression(C=0.5, tol=1e-10, max_iter=5000).fit(X, y)
    rs = RashomonSet(estimator="logistic", C=0.5 * n, fit_intercept=True, epsilon=0.01).fit(X, y)
    assert np.abs(rs.coef_ - sk.coef_.ravel()).max() < 1e-5
    assert abs(rs.intercept_ - sk.intercept_[0]) < 1e-5
    # the penalty mask leaves the intercept out of the Hessian's regularization
    H = rs._hessian_matrix()
    w = rs._w_diag
    assert H[0, 0] == pytest.approx(np.mean(w))  # no + lambda on the intercept
    assert rs._reg_mask[0] == 0.0 and rs._reg_mask[1:].all()


def test_penalize_intercept_flag_changes_solution(cancer):
    X, y = cancer
    a = RashomonSet(estimator="logistic", C=5.0, fit_intercept=True, epsilon=0.01).fit(X, y)
    b = RashomonSet(estimator="logistic", C=5.0, fit_intercept=True, epsilon=0.01, penalize_intercept=True).fit(X, y)
    assert abs(a.intercept_) > abs(b.intercept_)  # penalizing shrinks the intercept
    assert b._reg_mask[0] == 1.0
    assert a.get_params()["penalize_intercept"] is False


def test_ridge_and_ols_closed_form_match_sklearn(cancer):
    X, y01 = cancer
    rng = np.random.default_rng(0)
    y = X @ np.arange(1, 11) + rng.normal(size=len(y01)) + 5.0
    n = len(y)
    skr = Ridge(alpha=3.0).fit(X, y)
    rsr = RashomonSet(estimator="linear", C=n / 3.0, fit_intercept=True, epsilon=0.01).fit(X, y)
    assert np.abs(rsr.coef_ - skr.coef_).max() < 1e-9
    assert abs(rsr.intercept_ - skr.intercept_) < 1e-9
    skl = LinearRegression().fit(X, y)
    rsl = RashomonSet(estimator="linear", C=np.inf, fit_intercept=True, epsilon=0.01).fit(X, y)
    assert np.abs(rsl.coef_ - skl.coef_).max() < 1e-9
    assert rsl.diagnostics()["lambda"] == 0.0


def test_lambda_zero_logistic_and_separation_guard(cancer):
    rng = np.random.default_rng(1)
    X = rng.normal(size=(2000, 5))
    p = 1.0 / (1.0 + np.exp(-(X @ [1.0, -0.5, 0.3, 0.0, 0.2] + 0.3)))
    y = (rng.random(2000) < p).astype(float)
    sk = LogisticRegression(C=np.inf, tol=1e-10, max_iter=5000).fit(X, y)
    rs = RashomonSet(estimator="logistic", C=np.inf, fit_intercept=True, epsilon=0.01).fit(X, y)
    assert np.abs(rs.coef_ - sk.coef_.ravel()).max() < 1e-6
    # quasi-separable data has no finite unpenalized optimum -> refuse
    Xc, yc = cancer
    with pytest.raises(RuntimeError, match="unpenalized"):
        RashomonSet(estimator="logistic", C=np.inf, fit_intercept=True, epsilon=0.01).fit(Xc, yc)
    # ...but with a penalty the same data is fine (no spurious separation error)
    RashomonSet(estimator="logistic", C=0.5 * len(yc), fit_intercept=True, epsilon=0.01).fit(Xc, yc)


def test_invalid_C_and_labels(cancer):
    X, y = cancer
    with pytest.raises(ValueError, match="C must be > 0"):
        RashomonSet(estimator="logistic", C=0.0).fit(X, y)
    with pytest.raises(ValueError, match="y in \\{0, 1\\}"):
        RashomonSet(estimator="logistic").fit(X, 2 * y - 1)


def test_theta_init_and_newton_polish_reach_same_optimum(cancer):
    X, y = cancer
    n = len(y)
    cold = RashomonSet(estimator="logistic", C=0.5 * n, fit_intercept=True, epsilon=0.01).fit(X, y)
    warm = RashomonSet(estimator="logistic", C=0.5 * n, fit_intercept=True, epsilon=0.01).fit(
        X, y, theta_init=np.r_[0.3, np.zeros(10)]
    )
    assert np.abs(cold._theta_hat - warm._theta_hat).max() < 1e-6
    # gradient at the optimum is (numerically) zero
    p = 1.0 / (1.0 + np.exp(-(cold._X @ cold._theta_hat)))
    g = cold._X.T @ (p - y) / n + cold._lambda * cold._reg_mask * cold._theta_hat
    assert np.linalg.norm(g) < 1e-7
    with pytest.raises(ValueError, match="theta_init"):
        RashomonSet(estimator="logistic", fit_intercept=True).fit(X, y, theta_init=np.zeros(3))


def test_absolute_epsilon_mode(cancer):
    X, y = cancer
    rs = RashomonSet(estimator="logistic", epsilon=0.0042, epsilon_mode="absolute").fit(X, y)
    assert rs.diagnostics()["epsilon"] == pytest.approx(0.0042)
    with pytest.raises(ValueError):
        RashomonSet(estimator="logistic", epsilon=-1.0, epsilon_mode="absolute").fit(X, y)


def test_refit_resets_cached_operators(cancer):
    X, y = cancer
    rs = RashomonSet(estimator="logistic", epsilon=0.02).fit(X, y)
    rs._hessian_matrix()
    rs._hessian_cholesky()
    rs.fit(X[:, :4], y)
    assert rs._hessian_matrix().shape == (4, 4)
    assert rs._hessian_cholesky().shape == (4, 4)
    assert rs.sample_ellipsoid(5, random_state=0).shape == (5, 4)


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
