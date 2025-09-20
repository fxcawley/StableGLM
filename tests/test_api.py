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
    assert r2 > 0.8


