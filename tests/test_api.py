import numpy as np

from rashomon import RashomonSet


def _make_data(n: int = 40, d: int = 6, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    w = rng.normal(size=d)
    logits = X @ w
    p = 1.0 / (1.0 + np.exp(-logits))
    y = (rng.random(n) < p).astype(float)
    return X, y


def test_lr_alpha_bootstrap_fallback_runs():
    # Force penalized logistic (lambda>0) so Wilks preconditions are violated
    X, y = _make_data(n=60, d=6, seed=11)
    # Use epsilon_mode=LR_alpha with bootstrap_fallback enabled and few reps for speed
    rs = RashomonSet(estimator="logistic", C=2.0, epsilon=0.2, epsilon_mode="LR_alpha", bootstrap_fallback=True, bootstrap_reps=50, random_state=0)
    rs.fit(X, y)
    diag = rs.diagnostics()
    # epsilon should be set; implied_alpha reported
    assert diag["epsilon"] is not None
    assert diag["epsilon_mode"] == "LR_alpha"
    assert 0 < diag["implied_alpha"] < 1


def test_objective_and_contains():
    X, y = _make_data(n=50, d=5, seed=1)
    rs = RashomonSet(estimator="logistic", random_state=0).fit(X, y)

    # L_hat matches objective at theta_hat
    L_hat = rs.diagnostics()["L_hat"]
    L_hat_from_obj = rs.objective(rs.coef_)
    assert abs(L_hat - L_hat_from_obj) < 1e-6

    # contains(theta_hat) must be True
    assert rs.contains(rs.coef_)

    # A far-away parameter should not be contained
    e1 = np.zeros(rs.n_features_in_, dtype=float)
    e1[0] = 1.0
    far_theta = rs.coef_ + 10.0 * e1
    assert not rs.contains(far_theta)


def test_predictive_api_logistic():
    X, y = _make_data(n=40, d=4, seed=2)
    rs = RashomonSet(estimator="logistic", random_state=0).fit(X, y)

    # decision function shape and values
    scores = rs.decision_function(X)
    assert scores.shape == (X.shape[0],)

    # predict_proba well-formed probabilities
    proba = rs.predict_proba(X)
    assert proba.shape == (X.shape[0], 2)
    row_sums = proba.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-8)

    # predict produces 0/1 labels
    y_pred = rs.predict(X)
    assert y_pred.shape == (X.shape[0],)
    assert set(np.unique(y_pred)).issubset({0, 1})

    # sklearn-style attrs
    assert rs.coef_.shape == (X.shape[1],)
    assert rs.n_features_in_ == X.shape[1]

    # probability bands contain predicted probabilities' point estimate
    bands = rs.probability_bands(X[:5])
    assert bands.shape == (5, 2)
    # center probability using theta_hat must lie within bands
    scores5 = rs.decision_function(X[:5])
    p_center = 1.0 / (1.0 + np.exp(-scores5))
    assert np.all(bands[:, 0] <= p_center + 1e-12)
    assert np.all(p_center <= bands[:, 1] + 1e-12)


def test_predictive_api_linear():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(30, 3))
    true_w = np.array([1.0, -2.0, 0.5])
    y = X @ true_w + 0.1 * rng.normal(size=30)

    rs = RashomonSet(estimator="linear", random_state=0).fit(X, y)

    scores = rs.decision_function(X)
    preds = rs.predict(X)
    assert scores.shape == (X.shape[0],)
    assert np.allclose(scores, preds)
    # linear predictions equal X @ coef_
    assert np.allclose(preds, X @ rs.coef_, atol=1e-6)


def test_diagnostics_fields():
    X, y = _make_data(n=25, d=7, seed=4)
    rs = RashomonSet(random_state=0).fit(X, y)
    diag = rs.diagnostics()
    # New fields exposed
    assert "L_hat" in diag and diag["L_hat"] > 0
    assert "theta_norm_2" in diag and diag["theta_norm_2"] >= 0
    # Existing invariants
    assert diag["n"] == X.shape[0] and diag["d"] == X.shape[1]
    assert diag["measure"] in {"param", "lr"}


def test_vectorized_membership_matches_scalar():
    X, y = _make_data(n=40, d=5, seed=7)
    rs = RashomonSet(estimator="logistic", random_state=0).fit(X, y)

    rng = np.random.default_rng(0)
    Theta = rng.normal(size=(8, rs.n_features_in_))
    # include theta_hat ensure at least one True
    Theta[0] = rs.coef_

    gaps_many = rs.loss_gap_many(Theta)
    gaps_scalar = np.array([rs.loss_gap(t) for t in Theta])
    assert np.allclose(gaps_many, gaps_scalar, atol=1e-10)

    mask_many = rs.contains_many(Theta)
    mask_scalar = np.array([rs.contains(t) for t in Theta])
    assert np.array_equal(mask_many, mask_scalar)


def test_membership_with_precomputed_margins():
    X, y = _make_data(n=30, d=4, seed=9)
    rs = RashomonSet(estimator="logistic", random_state=0).fit(X, y)

    theta_hat = rs.coef_
    margins_hat = X @ theta_hat

    # objective / loss gap agree when margins are precomputed
    obj_direct = rs.objective(theta_hat)
    obj_cached = rs.objective(theta_hat, x_theta=margins_hat)
    assert np.allclose(obj_direct, obj_cached, atol=1e-12)

    gap_direct = rs.loss_gap(theta_hat)
    gap_cached = rs.loss_gap(theta_hat, x_theta=margins_hat)
    assert np.allclose(gap_direct, gap_cached, atol=1e-12)

    assert rs.contains(theta_hat, x_theta=margins_hat)

    Theta = np.vstack([theta_hat, theta_hat + 0.05 * np.ones_like(theta_hat)])
    margins = X @ Theta.T  # shape (n, k)

    obj_many = rs.objective_many(Theta)
    obj_many_cached = rs.objective_many(Theta, XTheta=margins)
    assert np.allclose(obj_many, obj_many_cached, atol=1e-12)

    gaps_many = rs.loss_gap_many(Theta)
    gaps_many_cached = rs.loss_gap_many(Theta, XTheta=margins)
    assert np.allclose(gaps_many, gaps_many_cached, atol=1e-12)

    # Also accept transposed margins (k, n)
    margins_transposed = margins.T
    mask = rs.contains_many(Theta)
    mask_cached = rs.contains_many(Theta, XTheta=margins)
    mask_transposed = rs.contains_many(Theta, XTheta=margins_transposed)
    assert np.array_equal(mask, mask_cached)
    assert np.array_equal(mask, mask_transposed)


def test_hacking_interval_linear_bounds():
    # For linear model, hacking_interval on s should bound s^T theta
    rng = np.random.default_rng(10)
    X = rng.normal(size=(60, 6))
    true_w = rng.normal(size=6)
    y = X @ true_w + 0.1 * rng.normal(size=60)
    rs = RashomonSet(estimator="linear", random_state=0).fit(X, y)

    s = rng.normal(size=6)
    interval = rs.hacking_interval(s)
    val = float(s @ rs.coef_)
    assert interval["min"] <= val <= interval["max"]


def test_sklearn_params_and_score():
    X, y = _make_data(n=50, d=4, seed=12)
    rs = RashomonSet(estimator="logistic", random_state=0).fit(X, y)
    params = rs.get_params()
    assert params["estimator"] == "logistic" and "C" in params

    # set_params round-trip
    rs.set_params(C=2.0)
    assert rs.C == 2.0

    # score returns accuracy in [0,1]
    acc = rs.score(X, y)
    assert 0.0 <= acc <= 1.0

    # linear R^2 sanity
    rng = np.random.default_rng(13)
    Xl = rng.normal(size=(60, 3))
    w = np.array([1.0, -1.0, 0.5])
    yl = Xl @ w + 0.05 * rng.normal(size=60)
    rs_l = RashomonSet(estimator="linear", random_state=0).fit(Xl, yl)
    r2 = rs_l.score(Xl, yl)
    assert r2 > 0.7


def test_coef_intervals_and_sampler_membership():
    X, y = _make_data(n=50, d=5, seed=15)
    rs = RashomonSet(estimator="logistic", random_state=0, epsilon=0.05, epsilon_mode="percent_loss").fit(X, y)

    ivals = rs.coef_intervals()
    assert ivals.shape == (rs.n_features_in_, 2)
    # theta_hat coordinates lie within intervals for e_j dot theta_hat
    for j in range(rs.n_features_in_):
        val = rs.coef_[j]
        assert ivals[j, 0] <= val <= ivals[j, 1]

    # Sampling yields parameters inside the H-ellipsoid: (θ-θ̂)^T H (θ-θ̂) ≤ 2ε
    samp = rs.sample_ellipsoid(n_samples=20)
    H = rs._hessian_matrix()
    dtheta = samp - rs.coef_
    qvals = np.einsum("ni,ij,nj->n", dtheta, H, dtheta)
    eps = rs.diagnostics()["epsilon"]
    assert np.all(qvals <= 2.0 * float(eps) + 1e-8)

    # Hit-and-Run sampler basic smoke
    rs_whitened = RashomonSet(estimator="logistic", random_state=0, epsilon=0.05, epsilon_mode="percent_loss", sampler="hitandrun").fit(X, y)
    chain = rs_whitened.sample(n_samples=25, burnin=10, thin=2, random_state=123)
    assert chain.shape == (25, rs_whitened.n_features_in_)
    assert np.all(rs_whitened.contains_many(chain))

    # Explicit Euclidean directions without preconditioning
    rs_euclid = RashomonSet(
        estimator="logistic",
        random_state=0,
        epsilon=0.05,
        epsilon_mode="percent_loss",
        sampler="hitandrun",
    ).fit(X, y)
    chain_euclid = rs_euclid.sample(n_samples=15, burnin=5, thin=1, directions="euclidean", precondition=False, random_state=321)
    assert chain_euclid.shape == (15, rs_euclid.n_features_in_)
    assert np.all(rs_euclid.contains_many(chain_euclid))


def test_model_class_reliance_detects_informative_features():
    rng = np.random.default_rng(21)
    n, d = 80, 6

    # Linear: feature 0 strong signal
    Xl = rng.normal(size=(n, d))
    ylin = 3.0 * Xl[:, 0] + 0.1 * rng.normal(size=n)
    rs_l = RashomonSet(estimator="linear", random_state=0).fit(Xl, ylin)
    mcr_l = rs_l.model_class_reliance(Xl, ylin, n_permutations=8, random_state=0)
    imp_l = mcr_l["feature_importance"]
    assert imp_l[0] == np.max(imp_l)

    # Logistic: feature 1 strong signal
    Xg = rng.normal(size=(n, d))
    w = np.zeros(d)
    w[1] = 4.0
    logits = Xg @ w
    p = 1.0 / (1.0 + np.exp(-logits))
    ybin = (rng.random(n) < p).astype(float)
    rs_g = RashomonSet(estimator="logistic", random_state=0).fit(Xg, ybin)
    mcr_g = rs_g.model_class_reliance(Xg, ybin, n_permutations=8, random_state=0)
    imp_g = mcr_g["feature_importance"]
    assert imp_g[1] == np.max(imp_g)


def test_stationarity_at_theta_hat():
    """Verify theta_hat is a stationary point of the stated objective.

    This catches the sklearn regularization mismatch (C mapping).
    The gradient of L(theta) = (1/n)*sum(loss) + (lam/2)*||theta||^2
    must be ~0 at theta_hat.
    """
    X, y = _make_data(n=80, d=5, seed=42)
    rs = RashomonSet(
        estimator="logistic", C=1.0, random_state=0
    ).fit(X, y)

    theta = rs._theta_hat
    lam = rs._lambda
    n = rs._n
    z = rs._X @ theta
    p = 1.0 / (1.0 + np.exp(-z))
    grad = (rs._X.T @ (p - rs._y)) / n + lam * theta
    assert np.linalg.norm(grad) < 1e-3, (
        f"Gradient norm at theta_hat is {np.linalg.norm(grad):.4f}, "
        "expected ~0. Check sklearn C mapping."
    )


def test_stationarity_linear():
    """Verify theta_hat is optimal for ridge regression."""
    rng = np.random.default_rng(42)
    n, d = 80, 5
    X = rng.normal(size=(n, d))
    y = X @ rng.normal(size=d) + 0.1 * rng.normal(size=n)
    rs = RashomonSet(
        estimator="linear", C=1.0, random_state=0
    ).fit(X, y)

    theta = rs._theta_hat
    lam = rs._lambda
    # Gradient: -(1/n)*X^T(y - X*theta) + lam*theta
    grad = -(rs._X.T @ (rs._y - rs._X @ theta)) / n + lam * theta
    assert np.linalg.norm(grad) < 1e-6, (
        f"Ridge gradient norm is {np.linalg.norm(grad):.4e}. "
        "Check sklearn alpha mapping."
    )


def test_sklearn_vs_fallback_consistency():
    """Verify sklearn solver and fallback GD give the same theta_hat.

    This catches C/alpha mapping bugs since the fallback uses our
    stated objective directly.
    """
    X, y = _make_data(n=60, d=4, seed=99)

    # Fit with sklearn
    rs_sk = RashomonSet(
        estimator="logistic", C=2.0, max_iter=5000, tol=1e-10,
        random_state=0
    ).fit(X, y)

    # Fit with fallback (temporarily disable sklearn)
    import rashomon.rashomon_set as _mod
    orig = _mod._HAS_SK
    try:
        _mod._HAS_SK = False
        rs_fb = RashomonSet(
            estimator="logistic", C=2.0, max_iter=5000, tol=1e-10,
            random_state=0
        ).fit(X, y)
    finally:
        _mod._HAS_SK = orig

    np.testing.assert_allclose(
        rs_sk._theta_hat, rs_fb._theta_hat, atol=0.05,
        err_msg="sklearn and fallback solvers disagree on theta_hat"
    )
    np.testing.assert_allclose(
        rs_sk._L_hat, rs_fb._L_hat, atol=0.01,
        err_msg="sklearn and fallback solvers disagree on L_hat"
    )


def test_lr_alpha_highdim():
    """Test Sur-Candès high-dimensional LR calibration."""
    X, y = _make_data(n=80, d=10, seed=42)
    rs = RashomonSet(
        estimator="logistic",
        epsilon=0.05,
        epsilon_mode="LR_alpha_highdim",
        safety_override=True,
        random_state=0,
    ).fit(X, y)

    diag = rs.diagnostics()
    assert diag["epsilon_mode"] == "LR_alpha_highdim"
    assert diag["epsilon"] is not None
    assert diag["epsilon"] > 0
    # High-dim correction should produce a larger epsilon than low-dim
    # (correction factor > 1 for d/n > 0)
    assert diag["implied_alpha"] is not None


def test_lanczos_hinvsqrt():
    """Test Lanczos H^{-1/2} approximation."""
    X, y = _make_data(n=60, d=5, seed=42)
    rs = RashomonSet(estimator="logistic", random_state=0).fit(X, y)

    rng = np.random.default_rng(0)
    v = rng.normal(size=5)
    result = rs._lanczos_hinvsqrt_v(v, m=20)

    # Result should have the right shape
    assert result.shape == (5,)
    # H^{-1/2} v should satisfy: H (H^{-1/2} v) = H^{1/2} v
    # Verify: ||H^{1/2} (H^{-1/2} v)||^2 ≈ v^T H^{-1} H v = v^T v
    Hresult = rs._Hv(result)
    # H * H^{-1/2} v should be H^{1/2} v
    # Check that the norm is reasonable (not zero, not huge)
    assert np.linalg.norm(result) > 0
    assert np.linalg.norm(result) < 100 * np.linalg.norm(v)


def test_mcr_analytic():
    """Test analytic MCR bounds via quadratic relaxation."""
    X, y = _make_data(n=60, d=4, seed=42)
    rs = RashomonSet(
        estimator="logistic",
        epsilon=0.05,
        epsilon_mode="percent_loss",
        random_state=0,
    ).fit(X, y)

    result = rs.mcr_analytic(X, y, n_permutations=10, random_state=42)

    assert "mcr_min_analytic" in result
    assert "mcr_max_analytic" in result
    assert "importance_at_hat" in result
    assert "grad_norms" in result
    assert result["mcr_min_analytic"].shape == (4,)
    assert result["mcr_max_analytic"].shape == (4,)
    # Analytic bounds should bracket the point estimate
    assert np.all(result["mcr_min_analytic"] <= result["importance_at_hat"] + 1e-10)
    assert np.all(result["importance_at_hat"] <= result["mcr_max_analytic"] + 1e-10)
    # importance_at_hat measures loss increase from permutation -- should be >= 0 on average
    assert np.mean(result["importance_at_hat"]) >= -0.01
    # grad_norms should be non-negative (norms are always >= 0)
    assert np.all(result["grad_norms"] >= -1e-10)


def test_shapley_vic():
    """Test Shapley-VIC computation."""
    X, y = _make_data(n=60, d=4, seed=42)
    rs = RashomonSet(
        estimator="logistic",
        epsilon=0.05,
        epsilon_mode="percent_loss",
        random_state=0,
    ).fit(X, y)

    result = rs.shapley_vic(
        X, n_samples=30, feature_names=["a", "b", "c", "d"], random_state=42
    )

    assert "shapley_samples" in result
    assert "shapley_mean" in result
    assert "shapley_std" in result
    assert "shapley_min" in result
    assert "shapley_max" in result
    assert result["shapley_samples"].shape == (30, 4)
    assert result["shapley_mean"].shape == (4,)
    assert result["feature_names"] == ["a", "b", "c", "d"]
    # Shapley values should be non-negative (mean absolute)
    assert np.all(result["shapley_mean"] >= 0)
    # Min <= mean <= max
    assert np.all(result["shapley_min"] <= result["shapley_mean"] + 1e-10)
    assert np.all(result["shapley_mean"] <= result["shapley_max"] + 1e-10)


def test_capacity_delta_covering():
    """Test capacity with delta-covering number."""
    X, y = _make_data(n=60, d=4, seed=42)
    rs = RashomonSet(
        estimator="logistic",
        epsilon=0.05,
        epsilon_mode="percent_loss",
        random_state=0,
    ).fit(X, y)

    # Without delta
    cap = rs.capacity()
    assert "log_volume" in cap
    assert "effective_dim" in cap
    assert "log_covering_number" not in cap

    # With delta
    cap_d = rs.capacity(delta=0.01)
    assert "log_covering_number" in cap_d
    assert "delta" in cap_d
    assert cap_d["log_covering_number"] >= 0
    assert cap_d["delta"] == 0.01

    # Smaller delta should give larger covering number
    cap_d2 = rs.capacity(delta=0.001)
    assert cap_d2["log_covering_number"] >= cap_d["log_covering_number"]


def test_discrepancy_extremal_pair():
    """Test that discrepancy returns the extremal pair bound."""
    X, y = _make_data(n=60, d=4, seed=42)
    rs = RashomonSet(
        estimator="logistic",
        epsilon=0.05,
        epsilon_mode="percent_loss",
        random_state=0,
    ).fit(X, y)

    disc = rs.discrepancy(X, n_samples=50, n_pairs=50, random_state=42)
    assert "discrepancy_extremal_pair" in disc
    # Extremal pair should be <= the naive bound
    if disc["discrepancy_extremal_pair"] is not None:
        assert disc["discrepancy_extremal_pair"] <= disc["discrepancy_bound"] + 1e-6


def test_from_ensemble():
    """Test from_ensemble class method."""
    rng = np.random.default_rng(42)
    n, d = 50, 4
    X = rng.normal(size=(n, d))
    w = rng.normal(size=d)
    p = 1.0 / (1.0 + np.exp(-(X @ w)))
    y = (rng.random(n) < p).astype(float)

    # Create an ensemble of slightly different models
    models = []
    for i in range(20):
        theta = w + 0.1 * rng.normal(size=d)
        models.append(theta)

    result = RashomonSet.from_ensemble(
        models, X, y, estimator="logistic", lam=0.5,
        feature_names=["f0", "f1", "f2", "f3"]
    )

    assert result["n_models"] == 20
    assert result["coef_matrix"].shape == (20, 4)
    assert result["vic_mean"].shape == (4,)
    assert result["vic_std"].shape == (4,)
    assert result["vic_intervals"].shape == (4, 2)
    assert result["losses"].shape == (20,)
    assert 0 <= result["ambiguity_rate"] <= 1
    assert 0 <= result["max_pair_disagreement"] <= 1
    assert result["feature_names"] == ["f0", "f1", "f2", "f3"]

    # With lam>0, losses should include regularization (larger than data-only)
    result_noreg = RashomonSet.from_ensemble(
        models, X, y, estimator="logistic", lam=0.0,
    )
    # Regularized losses should be >= data-only losses
    assert np.all(result["losses"] >= result_noreg["losses"] - 1e-10)


def test_compare_to_bayesian():
    """Test Bayesian posterior comparison."""
    X, y = _make_data(n=80, d=4, seed=42)
    rs = RashomonSet(
        estimator="logistic",
        epsilon=0.05,
        epsilon_mode="percent_loss",
        random_state=0,
    ).fit(X, y)

    result = rs.compare_to_bayesian(
        n_samples=50,
        confidence=0.90,
        feature_names=["a", "b", "c", "d"],
        random_state=42,
    )

    assert "bayesian_samples" in result
    assert "bayesian_ci" in result
    assert "bayesian_std" in result
    assert "vic_samples" in result
    assert "vic_intervals" in result
    assert "comparison" in result
    assert "epsilon_vs_chi2" in result

    assert result["bayesian_samples"].shape == (50, 4)
    assert result["bayesian_ci"].shape == (4, 2)
    assert result["vic_intervals"].shape == (4, 2)

    # Bayesian CIs should bracket theta_hat
    for j in range(4):
        assert result["bayesian_ci"][j, 0] <= result["theta_hat"][j]
        assert result["theta_hat"][j] <= result["bayesian_ci"][j, 1]

    # Comparison should have all features
    assert len(result["comparison"]) == 4
    for name in ["a", "b", "c", "d"]:
        c = result["comparison"][name]
        assert "bayesian_width" in c
        assert "vic_width" in c
        assert "vic_to_bayesian_ratio" in c
        assert c["bayesian_width"] > 0
        assert c["vic_width"] > 0


def test_fit_intercept_logistic():
    """Test that fit_intercept=True works for logistic regression."""
    X, y = _make_data(n=80, d=4, seed=42)
    rs = RashomonSet(
        estimator="logistic", fit_intercept=True, random_state=0
    ).fit(X, y)

    # coef_ should have d features (not d+1)
    assert rs.coef_.shape == (4,)
    assert isinstance(rs.intercept_, float)
    assert rs.n_features_in_ == 4

    # Internal theta has d+1 dimensions
    assert rs._theta_hat.shape == (5,)
    assert rs._d == 5

    # Predict should accept original X shape
    preds = rs.predict(X)
    assert preds.shape == (80,)

    # Ambiguity should work
    amb = rs.ambiguity(X, threshold_mode="fixed", threshold_value=0.5)
    assert 0 <= amb["ambiguity_rate"] <= 1

    # Score
    acc = rs.score(X, y)
    assert 0 <= acc <= 1


def test_fit_intercept_linear():
    """Test fit_intercept for linear regression."""
    rng = np.random.default_rng(42)
    n, d = 80, 3
    X = rng.normal(size=(n, d))
    y = 2.0 + X @ np.array([1.0, -1.0, 0.5]) + 0.1 * rng.normal(size=n)

    # Use weak regularization so intercept is not shrunk to zero
    rs = RashomonSet(
        estimator="linear", fit_intercept=True, C=100.0, random_state=0
    ).fit(X, y)

    assert rs.coef_.shape == (3,)
    assert rs.n_features_in_ == 3
    # Intercept should be in the right ballpark (regularization shrinks it)
    assert abs(rs.intercept_) > 0.1
    # R^2 should be reasonable
    assert rs.score(X, y) > 0.5


