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


