"""Tests for the one-call ``audit`` API and the scikit-learn adapter."""

import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression,
    LogisticRegressionCV,
    Ridge,
    RidgeCV,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from rashomon import RashomonSet, StabilityReport, audit

# --------------------------------------------------------------------------- data


@pytest.fixture(scope="module")
def cancer():
    data = load_breast_cancer(as_frame=True)
    X = data.data.iloc[:, :8]
    X = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)
    return X, data.target


@pytest.fixture(scope="module")
def regression_data():
    rng = np.random.default_rng(0)
    n, d = 400, 6
    X = rng.normal(size=(n, d))
    X[:, 1] = X[:, 0] + 0.1 * rng.normal(size=n)  # a near-duplicate feature
    y = 2.0 + X @ np.array([1.0, 0.0, -0.5, 0.0, 0.3, 0.0]) + rng.normal(scale=1.0, size=n)
    return pd.DataFrame(X, columns=[f"f{j}" for j in range(d)]), pd.Series(y, name="target")


def _fast(**kw):
    """Small, deterministic audit settings for tests."""
    base = dict(n_samples=200, random_state=0)
    base.update(kw)
    return base


# ------------------------------------------------------------ objective reproduction


def test_logistic_reproduces_sklearn_optimum(cancer):
    X, y = cancer
    model = LogisticRegression(C=0.7, tol=1e-10, max_iter=5000).fit(X, y)
    report = audit(model, X, y, tolerance=0.02, **_fast())
    assert report.details["coef_max_abs_diff_vs_model"] < 1e-4
    # the user's own model must sit (numerically) at the optimum and inside the set
    assert report.details["model_loss_gap"] < 1e-8
    assert report.rashomon_set.contains(np.r_[model.intercept_, model.coef_.ravel()])
    assert report.details["lambda"] == pytest.approx(1.0 / (0.7 * len(y)))
    assert report.details["notes"] == []


def test_logistic_without_intercept(cancer):
    X, y = cancer
    model = LogisticRegression(C=1.0, fit_intercept=False, tol=1e-10, max_iter=5000).fit(X, y)
    report = audit(model, X, y, tolerance=0.02, **_fast())
    assert report.intercept is None and report.intercept_range is None
    assert report.details["coef_max_abs_diff_vs_model"] < 1e-4
    assert report.samples.shape[1] == X.shape[1]


def test_unpenalized_logistic_lambda_zero():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(1500, 4))
    p = 1.0 / (1.0 + np.exp(-(X @ [1.0, -0.5, 0.0, 0.3] + 0.2)))
    y = (rng.random(1500) < p).astype(int)
    model = LogisticRegression(C=np.inf, tol=1e-10, max_iter=5000).fit(X, y)
    report = audit(model, X, y, tolerance="lr", **_fast())
    assert report.details["lambda"] == 0.0
    assert report.details["coef_max_abs_diff_vs_model"] < 1e-4
    assert report.details["epsilon_mode"] == "LR_alpha"


def test_ridge_and_ols_reproduce_exactly(regression_data):
    X, y = regression_data
    ridge = Ridge(alpha=3.0).fit(X, y)
    report = audit(ridge, X, y, tolerance=0.05, threshold=None, **_fast())
    assert report.task == "regression"
    assert report.details["coef_max_abs_diff_vs_model"] < 1e-8
    assert report.details["lambda"] == pytest.approx(3.0 / len(y))
    assert report.flip_rate is None and "flipped" not in report.prediction_ranges
    ols = LinearRegression().fit(X, y)
    report_ols = audit(ols, X, y, tolerance=0.05, threshold=None, **_fast())
    assert report_ols.details["lambda"] == 0.0
    assert report_ols.details["coef_max_abs_diff_vs_model"] < 1e-8


def test_cv_variants_use_selected_strength(cancer):
    X, y = cancer
    lrcv = LogisticRegressionCV(Cs=[0.1, 1.0], cv=3, max_iter=2000).fit(X, y)
    report = audit(lrcv, X, y, tolerance=0.02, **_fast())
    assert report.details["lambda"] == pytest.approx(1.0 / (float(lrcv.C_[0]) * len(y)))
    rcv = RidgeCV(alphas=[0.1, 1.0, 10.0]).fit(X, y.astype(float))
    report_r = audit(rcv, X, y.astype(float), tolerance=0.02, threshold=None, **_fast())
    assert report_r.details["lambda"] == pytest.approx(float(rcv.alpha_) / len(y))


def test_from_sklearn_classmethod(cancer):
    X, y = cancer
    model = LogisticRegression(C=0.5, max_iter=5000, tol=1e-10).fit(X, y)
    rs = RashomonSet.from_sklearn(model, X, y, epsilon=0.02, random_state=0)
    assert rs.fit_intercept and not rs.penalize_intercept
    assert rs.coef_.shape == (X.shape[1],)
    assert np.abs(rs.coef_ - model.coef_.ravel()).max() < 1e-4
    with pytest.raises(ValueError, match="cannot be overridden"):
        RashomonSet.from_sklearn(model, X, y, C=1.0)


def test_sklearn_default_tolerance_is_accepted_silently(cancer):
    """sklearn's default tol=1e-4 stops ~1e-2 from the optimum in coefficient space, but the
    loss gap is negligible relative to any sensible tolerance; no note or warning should appear."""
    X, y = cancer
    model = LogisticRegression(max_iter=2000).fit(X, y)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        report = audit(model, X, y, tolerance=0.02, **_fast())
    assert report.details["coef_max_abs_diff_vs_model"] < 0.05
    assert report.details["model_loss_gap"] < 0.01 * report.tolerance
    assert not any("optimum" in n for n in report.details["notes"])


# ------------------------------------------------------------------ input handling


def test_pipeline_and_feature_names(cancer):
    X, y = cancer
    raw = load_breast_cancer(as_frame=True).data.iloc[:, :8]  # unscaled, let the pipeline scale
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, tol=1e-10)).fit(raw, y)
    report = audit(model, raw, y, tolerance=0.02, **_fast())
    assert list(report.coefficients.index) == list(raw.columns)
    assert report.details["coef_max_abs_diff_vs_model"] < 1e-4
    # new data goes through the same preprocessing
    new = report.predict_ranges(raw.iloc[:7])
    assert list(new.columns) == ["estimate", "low", "high", "flipped"]
    assert len(new) == 7
    pd.testing.assert_frame_equal(new, report.prediction_ranges.iloc[:7])


def test_numpy_inputs_get_generic_names(cancer):
    X, y = cancer
    model = LogisticRegression(max_iter=2000).fit(X.to_numpy(), y.to_numpy())
    report = audit(model, X.to_numpy(), y.to_numpy(), tolerance=0.02, **_fast())
    assert list(report.coefficients.index) == [f"x{j}" for j in range(X.shape[1])]
    report2 = audit(model, X.to_numpy(), y.to_numpy(), tolerance=0.02, feature_names=list("abcdefgh"), **_fast())
    assert list(report2.coefficients.index) == list("abcdefgh")
    with pytest.raises(ValueError, match="feature_names has length"):
        audit(model, X.to_numpy(), y.to_numpy(), tolerance=0.02, feature_names=["a"], **_fast())


def test_string_class_labels(cancer):
    X, y = cancer
    labels = np.where(y.to_numpy() == 1, "benign", "malignant")
    model = LogisticRegression(max_iter=5000, tol=1e-10).fit(X, labels)
    report = audit(model, X, labels, tolerance=0.02, **_fast())
    # positive class is classes_[1] == "malignant"; probabilities refer to it
    assert model.classes_[1] == "malignant"
    p_sklearn = model.predict_proba(X)[:, 1]
    assert np.allclose(report.prediction_ranges["estimate"], p_sklearn, atol=1e-3)


@pytest.mark.parametrize(
    "model, match",
    [
        (SVC(), "does not support"),
        (LogisticRegression(class_weight="balanced", max_iter=2000), "class_weight"),
        (LogisticRegression(penalty="l1", solver="liblinear"), "L2 or unpenalized"),
    ],
)
def test_unsupported_models_raise(cancer, model, match):
    X, y = cancer
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X, y)
    with pytest.raises((TypeError, ValueError), match=match):
        audit(model, X, y, tolerance=0.02, **_fast())


def test_multiclass_raises():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 3))
    y = rng.integers(0, 3, size=300)
    model = LogisticRegression(max_iter=2000).fit(X, y)
    with pytest.raises(ValueError, match="binary"):
        audit(model, X, y, tolerance=0.02, **_fast())


def test_unfitted_model_raises(cancer):
    X, y = cancer
    with pytest.raises(ValueError, match="fitted"):
        audit(LogisticRegression(), X, y)


# ------------------------------------------------------------------- tolerances


def test_tolerance_modes(cancer):
    X, y = cancer
    model = LogisticRegression(max_iter=2000).fit(X, y)
    cv_report = audit(model, X, y, **_fast())  # default "cv"
    assert cv_report.details["epsilon_mode"] == "absolute" and cv_report.tolerance > 0
    assert "cross-validated" in cv_report.tolerance_description
    rel = audit(model, X, y, tolerance=0.03, **_fast())
    assert rel.tolerance == pytest.approx(0.03 * rel.loss_optimum)
    absolute = audit(model, X, y, tolerance=("absolute", 0.004), **_fast())
    assert absolute.tolerance == pytest.approx(0.004)
    lr = audit(model, X, y, tolerance=("lr", 0.1), **_fast())
    assert lr.details["epsilon_mode"] == "LR_alpha"
    with pytest.raises(ValueError):
        audit(model, X, y, tolerance=1.5, **_fast())
    with pytest.raises(ValueError):
        audit(model, X, y, tolerance="nonsense", **_fast())


def test_larger_tolerance_flips_more(cancer):
    X, y = cancer
    model = LogisticRegression(max_iter=2000).fit(X, y)
    small = audit(model, X, y, tolerance=0.005, **_fast())
    large = audit(model, X, y, tolerance=0.08, **_fast())
    assert small.flip_rate <= large.flip_rate
    assert (large.coefficients["high"] - large.coefficients["low"]).mean() > (
        small.coefficients["high"] - small.coefficients["low"]
    ).mean()


# -------------------------------------------------------------- report invariants


def test_report_invariants(cancer):
    X, y = cancer
    model = LogisticRegression(max_iter=5000, tol=1e-10).fit(X, y)
    report = audit(model, X, y, tolerance=0.03, **_fast())
    assert isinstance(report, StabilityReport)
    n = len(y)
    assert report.flipped.shape == (n,) and report.flipped.dtype == bool
    assert report.n_flipped == report.flipped.sum()
    assert report.flip_rate == pytest.approx(report.flipped.mean())
    # one model cannot disagree on more rows than are flippable at all
    assert 0.0 <= report.max_disagreement <= report.flip_rate
    coef = report.coefficients
    assert list(coef.columns) == ["estimate", "low", "high", "sign_stable"]
    assert (coef["low"] <= coef["estimate"]).all() and (coef["estimate"] <= coef["high"]).all()
    assert (coef["sign_stable"] == ((coef["low"] > 0) | (coef["high"] < 0))).all()
    assert set(report.stable_features) | set(report.unstable_features) == set(coef.index)
    pr = report.prediction_ranges
    assert (pr["low"] <= pr["estimate"] + 1e-12).all() and (pr["estimate"] <= pr["high"] + 1e-12).all()
    assert (pr["flipped"].to_numpy() == report.flipped).all()
    assert np.allclose(pr["estimate"], model.predict_proba(X)[:, 1], atol=1e-3)
    # all sampled models really are in the set
    assert report.rashomon_set.contains_many(report.samples).all()
    assert report.n_models == report.samples.shape[0] == 200
    assert report.ess_min is not None and report.reliability
    assert report.runtime_seconds > 0


def test_exact_vs_sampled_coefficient_ranges(cancer):
    X, y = cancer
    model = LogisticRegression(max_iter=2000).fit(X, y)
    exact = audit(model, X, y, tolerance=0.03, **_fast())  # auto -> exact at this size
    assert exact.coefficient_ranges == "exact"
    sampled = audit(model, X, y, tolerance=0.03, exact_ranges=False, **_fast())
    assert sampled.coefficient_ranges == "sampled"
    assert any("sampled models" in n for n in sampled.details["notes"])
    # sampled ranges are inner approximations of the exact ones
    assert (sampled.coefficients["low"] >= exact.coefficients["low"] - 1e-9).all()
    assert (sampled.coefficients["high"] <= exact.coefficients["high"] + 1e-9).all()
    assert "exact range" in exact.summary() and "sampled models" in sampled.summary()
    assert exact.intercept_range[0] <= sampled.intercept_range[0] <= sampled.intercept_range[1] <= exact.intercept_range[1]
    with pytest.raises(ValueError, match="exact_ranges"):
        audit(model, X, y, tolerance=0.03, exact_ranges="yes", **_fast())


def test_flip_definition_matches_sampled_predictions(cancer):
    X, y = cancer
    model = LogisticRegression(max_iter=2000).fit(X, y)
    report = audit(model, X, y, tolerance=0.03, threshold=0.3, **_fast())
    rs = report.rashomon_set
    Xa = rs._prepare_X(X.to_numpy())
    tau = np.log(0.3 / 0.7)
    pred_hat = Xa @ rs._theta_hat > tau
    preds = (Xa @ report.samples.T) > tau
    expected = (preds != pred_hat[:, None]).any(axis=1)
    assert (report.flipped == expected).all()
    assert report.max_disagreement == pytest.approx((preds != pred_hat[:, None]).mean(axis=0).max())


def test_regression_threshold_enables_flips(regression_data):
    X, y = regression_data
    model = Ridge(alpha=1.0).fit(X, y)
    # default: no cutoff for regression, hence no flip analysis
    default = audit(model, X, y, tolerance=0.05, **_fast())
    assert default.threshold is None and default.flip_rate is None
    with pytest.raises(ValueError, match="threshold"):
        audit(model, X, y, tolerance=0.05, threshold="median", **_fast())
    report = audit(model, X, y, tolerance=0.05, threshold=float(y.median()), **_fast())
    assert report.flip_rate is not None and 0 < report.flip_rate < 1
    assert "flipped" in report.prediction_ranges
    # the near-duplicate feature pair should not both be sign-stable
    assert not (report.coefficients.loc["f0", "sign_stable"] and report.coefficients.loc["f1", "sign_stable"])


def test_summary_and_plot(cancer):
    X, y = cancer
    model = LogisticRegression(max_iter=2000).fit(X, y)
    report = audit(model, X, y, tolerance=0.03, **_fast())
    text = report.summary()
    assert "Flip under some equally-good model" in text
    assert "Sign stable" in text and "LogisticRegression" in text
    assert repr(report) == text
    import matplotlib

    matplotlib.use("Agg")
    fig = report.plot()
    assert len(fig.axes) == 2
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_ellipsoid_method(cancer):
    X, y = cancer
    model = LogisticRegression(max_iter=2000).fit(X, y)
    report = audit(model, X, y, tolerance=0.03, method="ellipsoid", **_fast(n_samples=400))
    assert "ellipsoid" in report.method
    assert report.ess_min is None
    assert report.rashomon_set.contains_many(report.samples).all()
    assert 10 <= report.n_models <= 400


def test_non_optimal_user_model_is_flagged(cancer):
    X, y = cancer
    # A badly under-converged solver: its coefficients are far from the optimum.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = LogisticRegression(max_iter=1, tol=1e-12).fit(X, y)
    with pytest.warns(UserWarning, match="outside the reconstructed Rashomon set"):
        report = audit(model, X, y, tolerance=0.001, **_fast())
    assert report.details["model_loss_gap"] > report.tolerance
