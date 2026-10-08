"""Translate fitted scikit-learn linear models into the objective audited by RashomonSet.

scikit-learn and :class:`~rashomon.RashomonSet` parameterize the same L2-penalized
objectives differently:

* ``LogisticRegression`` minimizes ``C * sum_i logloss_i + 0.5 * ||w||^2`` (intercept
  unpenalized). Dividing by ``C * n`` gives ``mean_i logloss_i + (1 / (2 C n)) ||w||^2``,
  so in mean-loss units ``lambda = 1 / (C * n)``.
* ``Ridge`` minimizes ``||y - Xw - b||^2 + alpha * ||w||^2``. Dividing by ``2n`` gives
  ``0.5 * mean_i (y_i - x_i w - b)^2 + (alpha / (2n)) ||w||^2``, so ``lambda = alpha / n``.
* ``LinearRegression`` and unpenalized ``LogisticRegression`` correspond to ``lambda = 0``.

This module is internal. The public entry points are :func:`rashomon.audit` and
:meth:`rashomon.RashomonSet.from_sklearn`.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import numpy as np
from scipy import sparse
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils.validation import check_is_fitted

Array = np.ndarray

SUPPORTED = (
    "LogisticRegression / LogisticRegressionCV (binary, L2 or unpenalized), "
    "Ridge / RidgeCV (single target), LinearRegression, or a Pipeline ending in one of these"
)


@dataclass(frozen=True)
class LinearModelSpec:
    """The audited objective, reconstructed from a fitted scikit-learn estimator."""

    estimator: str  # "logistic" | "linear"
    lam: float  # L2 strength in mean-loss units; 0 = unpenalized
    fit_intercept: bool
    theta: Array  # [intercept (if any), coef...] in RashomonSet layout
    classes: Optional[Array]  # logistic only: sklearn ``classes_`` (negative, positive)
    model_name: str
    n_features: int
    sample_weight: Optional[Array] = None  # effective per-row weights (sample_weight * class_weight), or None

    @property
    def C(self) -> float:
        """``RashomonSet``'s ``C`` (``1/lambda``; ``inf`` when unpenalized)."""
        return np.inf if self.lam == 0.0 else 1.0 / self.lam

    @property
    def loss_name(self) -> str:
        return "log-loss" if self.estimator == "logistic" else "squared error (½·MSE)"


MAX_DENSE_ELEMENTS = 20_000_000  # ~160 MB of float64


def to_numpy(a: Any, *, dtype: Any = float) -> Array:
    """Convert array-likes (pandas objects, scipy sparse matrices) to a numpy array.

    Sparse matrices are densified: the Hessian, the samplers and the exact-range
    solver all work with dense arrays. Very large sparse inputs are refused with an
    explanation rather than silently exhausting memory.
    """
    if sparse.issparse(a):
        n_elem = int(np.prod(a.shape))
        if n_elem > MAX_DENSE_ELEMENTS:
            raise ValueError(
                f"X is sparse with {a.shape[0]} x {a.shape[1]} = {n_elem:,} entries; the audit works on "
                f"dense arrays and densifies inputs up to {MAX_DENSE_ELEMENTS:,} entries. Audit a subsample "
                "of rows or reduce the number of features."
            )
        return np.asarray(a.toarray(), dtype=dtype)
    if hasattr(a, "to_numpy"):
        return np.asarray(a.to_numpy(), dtype=dtype)
    return np.asarray(a, dtype=dtype)


def check_feature_names(model: Any, X: Any) -> None:
    """Raise if ``X`` has column names and they differ from those the model was fitted on.

    scikit-learn records ``feature_names_in_`` when fitted on a DataFrame. Passing
    columns in a different order would silently audit the wrong model, so this is an
    error rather than a warning.
    """
    fitted_names = getattr(model, "feature_names_in_", None)
    if fitted_names is None or not hasattr(X, "columns"):
        return
    given = [str(c) for c in X.columns]
    expected = [str(c) for c in fitted_names]
    if given != expected:
        if sorted(given) == sorted(expected):
            hint = "same columns in a different order; pass X[model.feature_names_in_]"
        else:
            hint = f"expected {expected}, got {given}"
        raise ValueError(f"X's columns do not match the columns the model was fitted on ({hint}).")


def unwrap_pipeline(model: Any, X: Any) -> Tuple[Any, Any, Optional[List[str]]]:
    """Split a Pipeline into (final estimator, transformed X, transformed feature names).

    Non-pipeline models are returned unchanged with ``names=None``.
    """
    if not isinstance(model, Pipeline):
        return model, X, None
    if len(model.steps) == 1:
        return model.steps[-1][1], X, None
    pre = model[:-1]
    X_t = pre.transform(X)
    names: Optional[List[str]] = None
    try:
        names = [str(s) for s in pre.get_feature_names_out()]
    except Exception:
        names = None
    return model.steps[-1][1], X_t, names


def validate_sample_weight(sample_weight: Any, n_samples: int) -> Optional[Array]:
    """Return the user's ``sample_weight`` as a validated float array, or None."""
    if sample_weight is None:
        return None
    sw = to_numpy(sample_weight).ravel()
    if sw.shape != (n_samples,):
        raise ValueError(f"sample_weight must have shape ({n_samples},), got {sw.shape}.")
    if np.any(sw < 0) or not np.all(np.isfinite(sw)) or not (sw.sum() > 0):
        raise ValueError("sample_weight must be finite, non-negative and not all zero.")
    return sw


def effective_class_weights(model: Any, classes: Array, y: Array, sample_weight: Optional[Array]) -> Optional[Array]:
    """Per-row multipliers implied by ``model.class_weight`` (None when there is none).

    Mirrors scikit-learn, which multiplies each row's sample weight by the weight of
    its class (``compute_class_weight``; ``"balanced"`` uses the (weighted) class counts).
    """
    class_weight = getattr(model, "class_weight", None)
    if class_weight is None:
        return None
    try:
        cw = compute_class_weight(class_weight, classes=classes, y=y, sample_weight=sample_weight)
    except TypeError:  # scikit-learn < 1.4 has no sample_weight argument
        cw = compute_class_weight(class_weight, classes=classes, y=y)
    index = {label: k for k, label in enumerate(classes.tolist())}
    return np.asarray([cw[index[label]] for label in y.tolist()], dtype=float)


def describe_sklearn_model(model: Any, *, y: Any, sample_weight: Any = None) -> LinearModelSpec:
    """Reconstruct the penalized objective a fitted scikit-learn estimator minimized.

    ``y`` is the training target and ``sample_weight`` the weights passed to ``fit``
    (if any); both are needed because scikit-learn's regularization strength is
    relative to the *sum of weights* and ``class_weight`` multiplies them.
    """
    name = type(model).__name__
    try:
        check_is_fitted(model)
    except Exception as exc:  # NotFittedError
        raise ValueError(f"{name} must be fitted before auditing.") from exc
    y_arr = np.asarray(y.to_numpy() if hasattr(y, "to_numpy") else y).ravel()
    n_samples = y_arr.shape[0]
    sw = validate_sample_weight(sample_weight, n_samples)
    weight_sum = float(n_samples if sw is None else sw.sum())

    if isinstance(model, LogisticRegression):  # also LogisticRegressionCV
        classes = np.asarray(model.classes_)
        if classes.shape[0] != 2:
            raise ValueError(
                f"audit() supports binary classification; this {name} has {classes.shape[0]} classes. "
                "Multinomial support is planned."
            )
        cw_rows = effective_class_weights(model, classes, y_arr, sw)
        if cw_rows is not None:
            sw = cw_rows if sw is None else sw * cw_rows
            weight_sum = float(sw.sum())
        penalty = getattr(model, "penalty", "l2")
        l1_ratio = getattr(model, "l1_ratio", None)
        if hasattr(model, "l1_ratio_"):  # LogisticRegressionCV
            l1_ratio = np.ravel(model.l1_ratio_)[0]
        if penalty in ("l1", "elasticnet") or (
            l1_ratio is not None and not np.isnan(float(l1_ratio)) and float(l1_ratio) > 0.0
        ):
            raise ValueError(
                "Only L2 or unpenalized logistic regression is supported (the Rashomon set of an L1 / "
                "elastic-net objective is not a smooth convex level set)."
            )
        C = float(np.ravel(model.C_)[0]) if hasattr(model, "C_") else float(model.C)
        unpenalized = penalty in (None, "none") or not np.isfinite(C)
        # sklearn: C * sum_i w_i loss_i + 0.5||coef||^2; in weighted-mean-loss units lambda = 1/(C * sum w)
        lam = 0.0 if unpenalized else 1.0 / (C * weight_sum)
        if getattr(model, "solver", "") == "liblinear" and model.fit_intercept:
            warnings.warn(
                "solver='liblinear' penalizes the intercept (scaled by intercept_scaling); the audited "
                "objective leaves it unpenalized, so the reproduced coefficients may differ slightly.",
                stacklevel=3,
            )
        coef = np.ravel(model.coef_).astype(float)
        theta = np.concatenate([np.ravel(model.intercept_).astype(float), coef]) if model.fit_intercept else coef
        return LinearModelSpec("logistic", lam, bool(model.fit_intercept), theta, classes, name, coef.shape[0], sw)

    if isinstance(model, (Ridge, RidgeCV, LinearRegression)):
        coef = np.asarray(model.coef_, dtype=float)
        if coef.ndim != 1:
            raise ValueError("audit() supports a single regression target; got multi-output coef_.")
        if getattr(model, "positive", False):
            raise ValueError("positive=True adds sign constraints that audit() does not reconstruct.")
        if isinstance(model, LinearRegression):
            lam = 0.0
        else:
            alpha = model.alpha_ if isinstance(model, RidgeCV) else model.alpha
            alpha_arr = np.ravel(np.asarray(alpha, dtype=float))
            if alpha_arr.shape[0] != 1:
                raise ValueError("audit() requires a scalar Ridge alpha.")
            # sklearn: sum_i w_i r_i^2 + alpha||coef||^2; in 0.5*weighted-mean units lambda = alpha/sum w
            lam = float(alpha_arr[0]) / weight_sum
        theta = np.concatenate([[float(model.intercept_)], coef]) if model.fit_intercept else coef
        return LinearModelSpec("linear", lam, bool(model.fit_intercept), theta, None, name, coef.shape[0], sw)

    raise TypeError(f"audit() does not support {name}. Supported: {SUPPORTED}.")


def encode_target(spec: LinearModelSpec, y: Any) -> Array:
    """Map ``y`` to the {0,1} floats (logistic) or floats (linear) that RashomonSet expects."""
    y_arr = np.asarray(y.to_numpy() if hasattr(y, "to_numpy") else y).ravel()
    if spec.estimator == "logistic":
        assert spec.classes is not None
        neg, pos = spec.classes
        is_pos = y_arr == pos
        if not np.all(is_pos | (y_arr == neg)):
            raise ValueError(f"y contains labels other than the model's classes_ {list(spec.classes)}.")
        return is_pos.astype(float)
    return y_arr.astype(float)


def resolve_feature_names(
    estimator: Any,
    X: Any,
    pipeline_names: Optional[List[str]],
    n_features: int,
    feature_names: Optional[List[str]],
) -> List[str]:
    """Pick feature names: explicit > pipeline transformer names > estimator > DataFrame > x0..x{d-1}."""
    candidates: List[Optional[List[str]]] = [
        list(feature_names) if feature_names is not None else None,
        pipeline_names,
        [str(s) for s in estimator.feature_names_in_] if hasattr(estimator, "feature_names_in_") else None,
        [str(c) for c in X.columns] if hasattr(X, "columns") else None,
    ]
    for names in candidates:
        if names is not None:
            if len(names) != n_features:
                if names is candidates[0]:
                    raise ValueError(f"feature_names has length {len(names)}, expected {n_features}.")
                continue
            return names
    return [f"x{j}" for j in range(n_features)]


def rashomon_set_from_sklearn(
    cls: Any, model: Any, X: Any, y: Any, *, sample_weight: Any = None, **kwargs: Any
) -> Any:
    """Implementation of :meth:`RashomonSet.from_sklearn`."""
    forbidden = {"estimator", "C", "fit_intercept", "penalize_intercept"} & set(kwargs)
    if forbidden:
        raise ValueError(f"{sorted(forbidden)} are derived from the fitted model and cannot be overridden.")
    check_feature_names(model, X)
    est, X_t, _ = unwrap_pipeline(model, X)
    X_arr = to_numpy(X_t)
    if X_arr.ndim != 2:
        raise ValueError("X must be 2-dimensional.")
    spec = describe_sklearn_model(est, y=y, sample_weight=sample_weight)
    if spec.n_features != X_arr.shape[1]:
        raise ValueError(f"model was fitted on {spec.n_features} features but X has {X_arr.shape[1]}.")
    rs = cls(estimator=spec.estimator, C=spec.C, fit_intercept=spec.fit_intercept, penalize_intercept=False, **kwargs)
    rs.fit(X_arr, encode_target(spec, y), theta_init=spec.theta, sample_weight=spec.sample_weight)
    return rs
