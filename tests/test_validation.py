"""Validation against an external published result and against brute force.

1. Profile-likelihood confidence intervals. For an unpenalized logistic fit, the
   range of coefficient j over the Rashomon set with eps = chi2_1(0.95) / (2n) is, by
   definition, the 95% profile-likelihood confidence interval: maximizing beta_j
   subject to L(theta) <= L_hat + eps minimizes the loss over the other coordinates,
   which is what profiling does. R's ``confint()`` for the UCLA graduate-admissions
   logit example (admit ~ gre + gpa + rank) is a widely reproduced published output,
   so ``coef_extremes()`` must reproduce it.

   Source: UCLA Statistical Consulting Group, "Logit Regression | R Data Analysis
   Examples", https://stats.oarc.ucla.edu/r/dae/logit-regression/ (data:
   https://stats.idre.ucla.edu/stat/data/binary.csv, 400 rows, vendored in
   tests/data/ucla_admissions.csv). Published values copied from the page.

2. Brute force in two dimensions. With d = 2 the Rashomon set can be gridded, which
   gives independent values for the coefficient ranges, the set of flippable rows and
   the moments of the uniform distribution on the set. The exact routines and the
   sampler are checked against them.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import brentq, minimize
from scipy.stats import chi2
from sklearn.linear_model import LogisticRegression

from rashomon import RashomonSet, audit

DATA = Path(__file__).parent / "data" / "ucla_admissions.csv"

# R: summary(mylogit)$coefficients[, "Estimate"] and confint(mylogit), as printed on the UCLA page.
R_COEF = {
    "(Intercept)": -3.989979,
    "gre": 0.002264,
    "gpa": 0.804038,
    "rank2": -0.675443,
    "rank3": -1.340204,
    "rank4": -1.551464,
}
R_CONFINT = {
    "(Intercept)": (-6.271620, -1.79255),
    "gre": (0.000138, 0.00444),
    "gpa": (0.160296, 1.46414),
    "rank2": (-1.300889, -0.05675),
    "rank3": (-2.027671, -0.67037),
    "rank4": (-2.400027, -0.75354),
}
NAMES = list(R_COEF)


@pytest.fixture(scope="module")
def admissions():
    df = pd.read_csv(DATA)
    X = np.column_stack(
        [df["gre"], df["gpa"], df["rank"] == 2, df["rank"] == 3, df["rank"] == 4]
    ).astype(float)
    return X, df["admit"].to_numpy(dtype=float)


def test_mle_matches_r(admissions):
    X, y = admissions
    rs = RashomonSet(estimator="logistic", C=np.inf, fit_intercept=True, epsilon=0.01).fit(X, y)
    theta = rs._theta_hat
    for j, name in enumerate(NAMES):
        assert theta[j] == pytest.approx(R_COEF[name], abs=2e-6), name


def test_coef_extremes_reproduce_r_profile_confint(admissions):
    X, y = admissions
    n = len(y)
    eps = chi2.ppf(0.95, df=1) / (2 * n)
    rs = RashomonSet(estimator="logistic", C=np.inf, fit_intercept=True, epsilon=eps, epsilon_mode="absolute").fit(X, y)
    extremes = rs.coef_extremes()
    # R's confint.glm profiles on a grid and interpolates with a spline; agreement to ~1e-4
    for j, name in enumerate(NAMES):
        lo, hi = R_CONFINT[name]
        assert extremes[j, 0] == pytest.approx(lo, abs=1e-4), f"{name} lower"
        assert extremes[j, 1] == pytest.approx(hi, abs=1e-4), f"{name} upper"


def test_coef_extremes_match_independent_profiling(admissions):
    """Independent check of functional_range: root-find the profile log-likelihood directly."""
    X, y = admissions
    n = len(y)
    Xa = np.column_stack([np.ones(n), X])
    eps = chi2.ppf(0.95, df=1) / (2 * n)
    rs = RashomonSet(estimator="logistic", C=np.inf, fit_intercept=True, epsilon=eps, epsilon_mode="absolute").fit(X, y)
    theta_hat, L_hat = rs._theta_hat, rs._L_hat

    def nll(theta):
        z = Xa @ theta
        return float(np.mean(np.logaddexp(0.0, z) - y * z))

    def profile(value, j):
        # minimise the loss over the other coordinates with coordinate j fixed
        others = [k for k in range(6) if k != j]

        def obj(free):
            theta = theta_hat.copy()
            theta[j] = value
            theta[others] = free
            return nll(theta)

        res = minimize(obj, theta_hat[others], method="BFGS", options={"gtol": 1e-10})
        return res.fun - (L_hat + eps)

    for j in (1, 2, 4):  # gre, gpa, rank3
        lo, hi = rs.functional_range(np.eye(6)[j])
        half = hi - theta_hat[j]
        lo_ind = brentq(profile, theta_hat[j] - 3 * half, theta_hat[j], args=(j,), xtol=1e-10)
        hi_ind = brentq(profile, theta_hat[j], theta_hat[j] + 3 * half, args=(j,), xtol=1e-10)
        assert lo == pytest.approx(lo_ind, abs=2e-6)
        assert hi == pytest.approx(hi_ind, abs=2e-6)


def test_audit_profile_tolerance_reproduces_r(admissions):
    X, y = admissions
    df = pd.DataFrame(X, columns=NAMES[1:])
    model = LogisticRegression(C=np.inf, tol=1e-10, max_iter=5000).fit(df, y)
    report = audit(model, df, y, tolerance=("profile", 0.05), n_samples=100, random_state=0)
    assert "profile-likelihood" in report.tolerance_description
    for name in NAMES[1:]:
        lo, hi = R_CONFINT[name]
        assert report.coefficients.loc[name, "low"] == pytest.approx(lo, abs=1e-4)
        assert report.coefficients.loc[name, "high"] == pytest.approx(hi, abs=1e-4)
    assert report.intercept_range[0] == pytest.approx(R_CONFINT["(Intercept)"][0], abs=1e-4)
    assert report.intercept_range[1] == pytest.approx(R_CONFINT["(Intercept)"][1], abs=1e-4)


# ---------------------------------------------------------------- brute force, d = 2


@pytest.fixture(scope="module")
def grid_problem():
    rng = np.random.default_rng(7)
    n = 300
    X = rng.normal(size=(n, 2))
    X[:, 1] = 0.6 * X[:, 0] + 0.8 * X[:, 1]  # correlated features
    p = 1.0 / (1.0 + np.exp(-(1.2 * X[:, 0] - 0.4 * X[:, 1])))
    y = (rng.random(n) < p).astype(float)
    rs = RashomonSet(estimator="logistic", C=5.0, epsilon=0.04, random_state=0).fit(X, y)
    # grid over a box that comfortably contains the set
    ex = rs.coef_extremes()
    pad = 0.5 * (ex[:, 1] - ex[:, 0])
    g0 = np.linspace(ex[0, 0] - pad[0], ex[0, 1] + pad[0], 701)
    g1 = np.linspace(ex[1, 0] - pad[1], ex[1, 1] + pad[1], 701)
    T0, T1 = np.meshgrid(g0, g1, indexing="ij")
    Theta = np.column_stack([T0.ravel(), T1.ravel()])
    inside = rs.contains_many(Theta)
    return rs, X, y, Theta[inside], (g0[1] - g0[0], g1[1] - g1[0])


def test_grid_coefficient_ranges(grid_problem):
    rs, X, y, inset, step = grid_problem
    ex = rs.coef_extremes()
    for j in range(2):
        assert ex[j, 0] == pytest.approx(inset[:, j].min(), abs=1.5 * step[j])
        assert ex[j, 1] == pytest.approx(inset[:, j].max(), abs=1.5 * step[j])


def test_grid_flips(grid_problem):
    rs, X, y, inset, step = grid_problem
    margins = X @ inset.T  # (n, n_inset)
    pred_hat = X @ rs._theta_hat > 0
    grid_flip = ((margins > 0) != pred_hat[:, None]).any(axis=1)
    exact_flip = rs.can_flip(X, 0.0)
    # rows can disagree only when the grid resolution matters: the extreme margin is
    # within one grid cell of the threshold
    clearance = np.abs(X) @ np.asarray(step)
    extreme = np.where(pred_hat, margins.min(axis=1), margins.max(axis=1))
    ambiguous_at_resolution = np.abs(extreme) <= clearance
    disagree = grid_flip != exact_flip
    assert np.all(ambiguous_at_resolution[disagree]), "exact flip test disagrees with the grid away from the resolution limit"
    assert disagree.mean() < 0.03
    # sampled flips are a subset of the grid flips (up to the same resolution caveat)
    S = rs.sample_hitandrun(n_samples=1500, burnin=100, random_state=1, ellipsoid_mix=0.5)
    sampled_flip = (((X @ S.T) > 0) != pred_hat[:, None]).any(axis=1)
    assert np.all(grid_flip[sampled_flip] | ambiguous_at_resolution[sampled_flip])


def test_grid_sampler_moments(grid_problem):
    rs, X, y, inset, step = grid_problem
    S = rs.sample_hitandrun(n_samples=6000, burnin=200, random_state=2, ellipsoid_mix=0.5)
    ess = rs._last_sample_diagnostics["ess_per_param"].min()
    assert ess > 500
    mean_g, cov_g = inset.mean(axis=0), np.cov(inset, rowvar=False)
    mean_s, cov_s = S.mean(axis=0), np.cov(S, rowvar=False)
    sd = np.sqrt(np.diag(cov_g))
    # Monte Carlo error on the mean is sd/sqrt(ESS) ~ 0.045 sd at ESS 500
    assert np.all(np.abs(mean_s - mean_g) / sd < 0.2)
    assert np.all(np.abs(np.sqrt(np.diag(cov_s)) / sd - 1.0) < 0.1)
    corr_g = cov_g[0, 1] / (sd[0] * sd[1])
    corr_s = cov_s[0, 1] / np.sqrt(cov_s[0, 0] * cov_s[1, 1])
    assert corr_s == pytest.approx(corr_g, abs=0.05)
