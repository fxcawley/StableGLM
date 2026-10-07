"""One-call stability audit for fitted scikit-learn linear models.

    from rashomon import audit
    report = audit(model, X, y)
    print(report.summary())
    report.plot()

The audit asks one question about a model you have already fitted: *would an
equally good model have told you something different?* "Equally good" means a
training loss within a tolerance of the optimum (the ε-Rashomon set); the report
translates the set into decision language -- which predictions can flip, how far
coefficients can move, which signs are stable.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import KFold, StratifiedKFold

from ._sklearn import (
    LinearModelSpec,
    describe_sklearn_model,
    encode_target,
    resolve_feature_names,
    to_numpy,
    unwrap_pipeline,
)
from .rashomon_set import RashomonSet, _sigmoid

__all__ = ["audit", "StabilityReport"]

Array = np.ndarray
Tolerance = Union[str, float, Tuple[str, float]]

# Effective-sample-size thresholds used to label how far to trust sampled estimates.
_ESS_RELIABLE = 100.0
_ESS_FAIR = 30.0


@dataclass
class StabilityReport:
    """Result of :func:`audit`.

    Attributes are plain-language; the corresponding literature terms are given in
    brackets so results can be cross-referenced with the papers.

    Attributes
    ----------
    flip_rate : float or None
        Fraction of rows whose predicted label changes under *some* equally-good
        model [ambiguity, Marx, Calmon & Ustun 2020]. ``None`` for regression
        without a ``threshold``.
    flipped : ndarray of bool, shape (n,)
        Row mask of those predictions; join it back to your DataFrame.
    max_disagreement : float or None
        Largest fraction of rows on which a single equally-good model disagrees with
        your model [discrepancy, Marx et al. 2020]. Always ``<= flip_rate``.
    coefficients : DataFrame indexed by feature
        ``estimate`` (your model), ``low``/``high`` (range across equally-good
        models) and ``sign_stable`` (the range excludes zero) [variable importance
        cloud, Dong & Rudin 2020; hacking intervals, Coker, Rudin & King 2021].
        When ``coefficient_ranges == "exact"`` the range is the solution of a convex
        program over the true set; otherwise it is the range over sampled models.
    prediction_ranges : DataFrame
        Per-row ``estimate`` (probability or fitted value), ``low``, ``high`` and
        ``flipped``, over the sampled models.
    tolerance : float
        The loss gap defining "equally good", in the units of the training loss.
    ess_min, reliability : float, str
        Minimum effective sample size across coefficients and its plain-language
        label (``reliable`` / ``fair`` / ``unreliable``).

    Quantities computed over sampled models (flip rate, disagreement, prediction
    ranges, and coefficient ranges when not exact) can only *understate* the truth,
    because every sampled model really is in the set but the set is not exhausted.
    Increase ``n_samples`` to tighten them.
    """

    model_name: str
    task: str
    n: int
    n_features: int
    feature_names: List[str]
    tolerance: float
    tolerance_description: str
    loss_name: str
    loss_optimum: float
    method: str
    n_models: int
    ess_min: Optional[float]
    reliability: str
    threshold: Optional[float]
    flip_rate: Optional[float]
    n_flipped: Optional[int]
    flipped: Array
    max_disagreement: Optional[float]
    coefficients: pd.DataFrame
    coefficient_ranges: str  # "exact" | "sampled"
    intercept: Optional[float]
    intercept_range: Optional[Tuple[float, float]]
    prediction_ranges: pd.DataFrame
    runtime_seconds: float
    _rs: RashomonSet = field(repr=False)
    _samples: Array = field(repr=False)
    details: Dict[str, Any] = field(default_factory=dict)

    # ----------------------------------------------------------------- views
    @property
    def rashomon_set(self) -> RashomonSet:
        """The underlying :class:`RashomonSet` (expert API)."""
        return self._rs

    @property
    def samples(self) -> Array:
        """Sampled parameter vectors, shape (n_models, d [+1 with intercept]); intercept first."""
        return self._samples

    @property
    def stable_features(self) -> List[str]:
        mask = self.coefficients["sign_stable"].to_numpy()
        return [str(f) for f in self.coefficients.index[mask]]

    @property
    def unstable_features(self) -> List[str]:
        mask = ~self.coefficients["sign_stable"].to_numpy()
        return [str(f) for f in self.coefficients.index[mask]]

    # ------------------------------------------------------------- new data
    def predict_ranges(self, X: Any) -> pd.DataFrame:
        """Prediction ranges (and flips) for new rows, using the sampled models."""
        pre = self.details.get("_preprocessor")
        X_t = pre.transform(X) if pre is not None else X
        X_arr = to_numpy(X_t)
        if X_arr.ndim != 2 or X_arr.shape[1] != self.n_features:
            raise ValueError(f"X must be 2-D with {self.n_features} features.")
        tau = _threshold_to_score(self.task, self.threshold)
        theta_hat = self._rs._theta_hat
        assert theta_hat is not None
        scan = _scan_predictions(self._rs._prepare_X(X_arr), theta_hat, self._samples, tau)
        return _ranges_frame(self.task, scan, tau, index=getattr(X, "index", None))

    # --------------------------------------------------------------- output
    def summary(self) -> str:
        """Plain-language summary. ``print(report.summary())`` or just ``report``."""
        w = 44
        lines = [
            f"Stability audit -- {self.model_name} ({self.task}), n={self.n:,}, {self.n_features} features",
            f"Equally-good models: training {self.loss_name} within {self.tolerance:.4g} of the optimum {self.loss_optimum:.4g}",
            f"  ({self.tolerance_description})",
            f"Method: {self.method}; {self.n_models:,} models"
            + (f", min ESS = {self.ess_min:.0f} -> {self.reliability}" if self.ess_min is not None else f" -> {self.reliability}"),
            "",
        ]
        if self.flip_rate is not None:
            lines += [
                "Predictions (over the sampled models; true values can only be higher)",
                f"  {'Flip under some equally-good model:':<{w}}{self.flip_rate:7.1%}  ({self.n_flipped:,} of {self.n:,})",
                f"  {'Worst single-model disagreement with yours:':<{w}}{self.max_disagreement:7.1%}",
                "",
            ]
        else:
            width = (self.prediction_ranges["high"] - self.prediction_ranges["low"]).median()
            resid_sd = self.details.get("residual_std")
            extra = f"  (residual std {resid_sd:.3g})" if resid_sd is not None else ""
            lines += ["Predictions (over the sampled models)", f"  {'Median prediction range width:':<{w}}{width:8.3g}{extra}", ""]
        coef = self.coefficients
        if self.coefficient_ranges == "exact":
            lines.append("Coefficients (exact range across all equally-good models)")
        else:
            lines.append("Coefficients (range across the sampled models; true range can only be wider)")
        name_w = max(8, min(32, max(len(str(i)) for i in coef.index)))
        lines.append(f"  {'feature':<{name_w}} {'estimate':>10} {'low':>10} {'high':>10}  sign")
        for name, row in coef.iterrows():
            flag = "stable" if row["sign_stable"] else "UNSTABLE"
            lines.append(f"  {str(name)[:name_w]:<{name_w}} {row['estimate']:>10.4g} {row['low']:>10.4g} {row['high']:>10.4g}  {flag}")
        n_stable = int(coef["sign_stable"].sum())
        lines.append(f"Sign stable: {n_stable} of {len(coef)} features." + (f" Unstable: {', '.join(self.unstable_features)}." if n_stable < len(coef) else ""))
        notes = self.details.get("notes") or []
        if notes:
            lines += [""] + [f"Note: {n}" for n in notes]
        return "\n".join(lines)

    def __repr__(self) -> str:  # noqa: D105
        return self.summary()

    def plot(self, max_features: int = 30, max_rows: int = 2000, figsize: Optional[Tuple[float, float]] = None) -> Any:
        """One figure: prediction ranges per row (left) and coefficient ranges (right)."""
        import matplotlib.pyplot as plt

        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=figsize or (13, 5))
        pr = self.prediction_ranges
        if len(pr) > max_rows:
            pr = pr.sample(max_rows, random_state=0)
        pr = pr.sort_values("estimate").reset_index(drop=True)
        x = np.arange(len(pr))
        if "flipped" in pr:
            colors = np.where(pr["flipped"], "#d62728", "#9e9e9e")
        else:
            colors = np.full(len(pr), "#9e9e9e")
        ax0.vlines(x, pr["low"], pr["high"], colors=colors, linewidth=0.8, alpha=0.8)
        ax0.plot(x, pr["estimate"], color="black", linewidth=1.0, label="your model")
        if self.threshold is not None:
            ax0.axhline(self.threshold, color="#1f77b4", linestyle="--", linewidth=1, label=f"threshold {self.threshold:g}")
            ax0.plot([], [], color="#d62728", linewidth=2, label=f"can flip ({self.flip_rate:.1%})")
        ax0.set_xlabel("rows, sorted by your model's prediction")
        ax0.set_ylabel("predicted probability" if self.task == "classification" else "fitted value")
        ax0.set_title("Prediction range across equally-good models")
        ax0.legend(loc="best", fontsize=8)

        coef = self.coefficients.copy()
        if len(coef) > max_features:
            coef = coef.loc[coef["estimate"].abs().sort_values(ascending=False).index[:max_features]]
        coef = coef.iloc[::-1]
        yy = np.arange(len(coef))
        cols = np.where(coef["sign_stable"], "#2ca02c", "#ff7f0e")
        ax1.hlines(yy, coef["low"], coef["high"], colors=cols, linewidth=3, alpha=0.8)
        ax1.scatter(coef["estimate"], yy, color="black", s=18, zorder=3)
        ax1.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax1.set_yticks(yy)
        ax1.set_yticklabels([str(i) for i in coef.index], fontsize=8)
        ax1.set_xlabel("coefficient")
        ax1.set_title("Coefficient range (green: sign stable, orange: can change sign)")
        fig.tight_layout()
        return fig


# ---------------------------------------------------------------------------
# audit()
# ---------------------------------------------------------------------------
def audit(
    model: Any,
    X: Any,
    y: Any,
    *,
    tolerance: Tolerance = "cv",
    threshold: Union[float, None, str] = "auto",
    n_samples: int = 1000,
    cv: int = 5,
    method: str = "auto",
    exact_ranges: Union[bool, str] = "auto",
    random_state: Optional[int] = None,
    feature_names: Optional[Sequence[str]] = None,
) -> StabilityReport:
    """Audit whether a fitted scikit-learn linear model's conclusions survive equally-good models.

    Parameters
    ----------
    model : fitted estimator
        ``LogisticRegression`` / ``LogisticRegressionCV`` (binary, L2 or unpenalized),
        ``Ridge`` / ``RidgeCV``, ``LinearRegression``, or a ``Pipeline`` ending in one
        of these (the preprocessing steps are applied to ``X`` for you).
    X, y : array-like or pandas
        The training data the model was fitted on. Feature names are taken from
        ``X.columns`` / the pipeline when available.
    tolerance : "cv" | float | "lr" | ("lr", alpha) | ("absolute", gap), default "cv"
        What "equally good" means:

        * ``"cv"`` -- a training-loss gap of one cross-validation standard error of
          the held-out loss (the one-standard-error rule familiar from glmnet).
          Data-driven; costs ``cv`` extra fits of your model.
        * a float ``r`` in (0, 1) -- a loss gap of ``r`` times the optimal training loss
          (``0.01`` = "models at most 1% worse").
        * ``"lr"`` / ``("lr", alpha)`` -- likelihood-ratio calibration: the set of
          models not rejected at level ``alpha`` (default 0.05). Exact for unpenalized
          fits, heuristic under regularization.
        * ``("absolute", gap)`` -- an explicit loss gap in training-loss units.
    threshold : float, None or "auto", default "auto"
        Decision threshold on the predicted probability (classification; ``"auto"``
        means 0.5, as in ``model.predict``) or on the fitted value (regression;
        ``"auto"`` means ``None``, i.e. no flip analysis unless you supply a cutoff).
    n_samples : int, default 1000
        Number of equally-good models to sample. More samples tighten ranges and
        raise the effective sample size.
    cv : int, default 5
        Folds used by ``tolerance="cv"``.
    method : "auto" | "sample" | "ellipsoid", default "auto"
        ``"sample"`` (what ``"auto"`` chooses) draws from the exact set with a
        hit-and-run chain accelerated by ellipsoid proposals. ``"ellipsoid"`` draws
        i.i.d. from the Hessian ellipsoid and keeps only draws that are truly in the
        set -- fast, but it cannot reach parts of the set outside the ellipsoid.
    exact_ranges : bool or "auto", default "auto"
        Compute coefficient ranges (and hence ``sign_stable``) exactly by convex
        optimization over the true set instead of from the sampled models. Costs a
        few dozen Hessian builds per coefficient; ``"auto"`` does it whenever
        ``n * n_features**2 <= 5e6`` (a few seconds at most).
    random_state : int, optional
    feature_names : sequence of str, optional
        Overrides inferred names.

    Returns
    -------
    StabilityReport
    """
    t0 = time.time()
    if method not in ("auto", "sample", "ellipsoid"):
        raise ValueError("method must be 'auto', 'sample' or 'ellipsoid'")
    if n_samples < 10:
        raise ValueError("n_samples must be at least 10")

    est, X_t, pipeline_names = unwrap_pipeline(model, X)
    X_arr = to_numpy(X_t)
    if X_arr.ndim != 2:
        raise ValueError("X must be 2-dimensional.")
    n = X_arr.shape[0]
    spec = describe_sklearn_model(est, n)
    if spec.n_features != X_arr.shape[1]:
        raise ValueError(f"model was fitted on {spec.n_features} features but X has {X_arr.shape[1]}.")
    names = resolve_feature_names(est, X, pipeline_names, spec.n_features, list(feature_names) if feature_names is not None else None)
    y_enc = encode_target(spec, y)
    task = "classification" if spec.estimator == "logistic" else "regression"
    if isinstance(threshold, str):
        if threshold != "auto":
            raise ValueError("threshold must be a number, None or 'auto'")
        threshold = 0.5 if task == "classification" else None
    notes: List[str] = []

    eps_value, eps_mode, eps_desc = _resolve_tolerance(tolerance, est, X_arr, y, y_enc, spec, cv, random_state)

    rs = RashomonSet(
        estimator=spec.estimator,
        C=spec.C,
        fit_intercept=spec.fit_intercept,
        penalize_intercept=False,
        epsilon=eps_value,
        epsilon_mode=eps_mode,
        sampler="hitandrun",
        random_state=random_state,
    ).fit(X_arr, y_enc, theta_init=spec.theta)
    assert rs._theta_hat is not None and rs._epsilon_value is not None and rs._L_hat is not None
    eps = float(rs._epsilon_value)

    # Did we reconstruct *their* model? Their coefficients should sit at our optimum.
    coef_diff = float(np.max(np.abs(spec.theta - rs._theta_hat)))
    user_gap = float(rs.objective(spec.theta) - rs._L_hat)
    if user_gap > eps:
        warnings.warn(
            f"The fitted model's coefficients are outside the reconstructed Rashomon set (loss gap "
            f"{user_gap:.3g} > tolerance {eps:.3g}). The training objective was probably not reproduced "
            "exactly (sample weights? a non-converged solver?). The audit is centred on the exact optimum "
            "of the reconstructed objective instead.",
            stacklevel=2,
        )
        notes.append("your model's coefficients lie outside the audited set; see warning")
    elif user_gap > 0.1 * eps:
        notes.append(
            f"your model's coefficients are {user_gap / eps:.0%} of the tolerance away from the exact optimum "
            "(solver tolerance); the audit is centred on the exact optimum"
        )

    # Sample equally-good models.
    use_ellipsoid = method == "ellipsoid"
    if use_ellipsoid:
        raw = rs.sample_ellipsoid(n_samples=n_samples, random_state=random_state)
        inside = rs.contains_many(raw)
        samples = raw[inside]
        fidelity = float(np.mean(inside))
        if samples.shape[0] < 10:
            raise RuntimeError(
                f"Only {samples.shape[0]} of {n_samples} ellipsoid draws fall inside the Rashomon set; "
                "use method='sample'."
            )
        diag = rs.compute_sample_diagnostics(samples, compute_ess=False, compute_isotropy=False)
        ess_min: Optional[float] = None
        method_desc = f"i.i.d. draws from the Hessian ellipsoid, {fidelity:.0%} inside the set"
        reliability = "approximate (ellipsoid only; cannot reach parts of the set outside it)"
    else:
        samples = rs.sample_hitandrun(n_samples=n_samples, burnin=200, random_state=random_state, ellipsoid_mix=0.5)
        diag = rs._last_sample_diagnostics or {}
        ess = diag.get("ess_per_param")
        ess_min = float(np.min(ess)) if ess is not None else None
        method_desc = "hit-and-run sampling of the exact set (with ellipsoid proposals)"
        reliability = _reliability_label(ess_min)
        if ess_min is not None and ess_min < _ESS_FAIR:
            notes.append(f"min ESS {ess_min:.0f} < {_ESS_FAIR:.0f}: increase n_samples or reduce the number of features")

    # Metrics on the training rows.
    tau = _threshold_to_score(task, threshold)
    scan = _scan_predictions(rs._prepare_X(X_arr), rs._theta_hat, samples, tau)
    scores_hat, flipped = scan["scores_hat"], scan["flipped"]
    if tau is not None:
        flip_rate: Optional[float] = float(np.mean(flipped))
        n_flipped: Optional[int] = int(np.sum(flipped))
        max_disagreement: Optional[float] = float(np.max(scan["disagree_counts"]) / n)
    else:
        flip_rate = n_flipped = max_disagreement = None

    offset = 1 if spec.fit_intercept else 0
    coef_hat = rs._theta_hat[offset:]
    if exact_ranges not in (True, False, "auto"):
        raise ValueError("exact_ranges must be True, False or 'auto'")
    do_exact = exact_ranges is True or (exact_ranges == "auto" and n * spec.n_features**2 <= 5e6)
    if do_exact:
        extremes = rs.coef_extremes()
        low, high = extremes[offset:, 0], extremes[offset:, 1]
        coefficient_ranges = "exact"
    else:
        low = samples[:, offset:].min(axis=0)
        high = samples[:, offset:].max(axis=0)
        coefficient_ranges = "sampled"
        notes.append("coefficient ranges are over sampled models (pass exact_ranges=True for exact ranges)")
    coefficients = pd.DataFrame(
        {"estimate": coef_hat, "low": np.minimum(low, coef_hat), "high": np.maximum(high, coef_hat)},
        index=pd.Index(names, name="feature"),
    )
    coefficients["sign_stable"] = (coefficients["low"] > 0) | (coefficients["high"] < 0)
    intercept = float(rs._theta_hat[0]) if spec.fit_intercept else None
    if spec.fit_intercept:
        intercept_range = (float(extremes[0, 0]), float(extremes[0, 1])) if do_exact else (
            float(samples[:, 0].min()),
            float(samples[:, 0].max()),
        )
    else:
        intercept_range = None

    prediction_ranges = _ranges_frame(task, scan, tau, index=X.index if hasattr(X, "index") else None)

    details: Dict[str, Any] = {
        "epsilon": eps,
        "epsilon_mode": eps_mode,
        "lambda": rs._lambda,
        "sampled_coefficient_ranges": np.c_[samples[:, offset:].min(axis=0), samples[:, offset:].max(axis=0)],
        "coef_max_abs_diff_vs_model": coef_diff,
        "model_loss_gap": user_gap,
        "sampler_diagnostics": diag,
        "notes": notes,
        "_preprocessor": model[:-1] if (hasattr(model, "steps") and len(model.steps) > 1) else None,
    }
    if task == "regression":
        details["residual_std"] = float(np.std(y_enc - scores_hat, ddof=1)) if n > 1 else None

    return StabilityReport(
        model_name=spec.model_name,
        task=task,
        n=n,
        n_features=spec.n_features,
        feature_names=names,
        tolerance=eps,
        tolerance_description=eps_desc,
        loss_name=spec.loss_name,
        loss_optimum=float(rs._L_hat),
        method=method_desc,
        n_models=int(samples.shape[0]),
        ess_min=ess_min,
        reliability=reliability,
        threshold=threshold,
        flip_rate=flip_rate,
        n_flipped=n_flipped,
        flipped=flipped,
        max_disagreement=max_disagreement,
        coefficients=coefficients,
        coefficient_ranges=coefficient_ranges,
        intercept=intercept,
        intercept_range=intercept_range,
        prediction_ranges=prediction_ranges,
        runtime_seconds=time.time() - t0,
        details=details,
        _rs=rs,
        _samples=samples,
    )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _threshold_to_score(task: str, threshold: Optional[float]) -> Optional[float]:
    """Decision threshold on the linear-predictor scale (logit for classification)."""
    if threshold is None:
        return None
    if task == "classification":
        if not (0.0 < threshold < 1.0):
            raise ValueError("threshold must be a probability in (0, 1) for classification")
        return float(np.log(threshold / (1.0 - threshold)))
    return float(threshold)


def _scan_predictions(
    Xa: Array, theta_hat: Array, samples: Array, tau: Optional[float], chunk: int = 4096
) -> Dict[str, Array]:
    """Per-row score ranges and flips over sampled models, in row chunks to bound memory."""
    n = Xa.shape[0]
    scores_hat = Xa @ theta_hat
    lo = np.empty(n)
    hi = np.empty(n)
    flipped = np.zeros(n, dtype=bool)
    disagree_counts = np.zeros(samples.shape[0], dtype=np.int64)
    for start in range(0, n, chunk):
        sl = slice(start, min(start + chunk, n))
        Z = Xa[sl] @ samples.T  # (rows, n_models)
        lo[sl] = Z.min(axis=1)
        hi[sl] = Z.max(axis=1)
        if tau is not None:
            disagree = (Z > tau) != (scores_hat[sl] > tau)[:, None]
            flipped[sl] = disagree.any(axis=1)
            disagree_counts += disagree.sum(axis=0)
    return {"scores_hat": scores_hat, "low": lo, "high": hi, "flipped": flipped, "disagree_counts": disagree_counts}


def _ranges_frame(task: str, scan: Dict[str, Array], tau: Optional[float], index: Any = None) -> pd.DataFrame:
    est, lo, hi = scan["scores_hat"], scan["low"], scan["high"]
    if task == "classification":
        est, lo, hi = _sigmoid(est), _sigmoid(lo), _sigmoid(hi)
    frame = pd.DataFrame({"estimate": est, "low": lo, "high": hi}, index=index)
    if tau is not None:
        frame["flipped"] = scan["flipped"]
    return frame


def _reliability_label(ess_min: Optional[float]) -> str:
    if ess_min is None:
        return "unknown (ESS unavailable)"
    if ess_min >= _ESS_RELIABLE:
        return "reliable"
    if ess_min >= _ESS_FAIR:
        return "fair (increase n_samples for tighter estimates)"
    return "unreliable (increase n_samples)"


def _heldout_loss(spec: LinearModelSpec, fitted: Any, X_te: Array, y_te: Array) -> float:
    """Mean data loss of a refitted model on held-out rows, in the audited objective's units."""
    if spec.estimator == "logistic":
        z = np.ravel(fitted.decision_function(X_te))
        return float(np.mean(np.logaddexp(0.0, z) - y_te * z))
    pred = np.ravel(fitted.predict(X_te))
    return float(0.5 * np.mean((y_te - pred) ** 2))


def cv_loss_standard_error(
    est: Any, X: Array, y_raw: Any, y_enc: Array, spec: LinearModelSpec, cv: int, random_state: Optional[int]
) -> Tuple[float, float]:
    """(standard error, mean) of the held-out loss across ``cv`` folds, refitting a clone of ``est``."""
    if cv < 2:
        raise ValueError("cv must be at least 2")
    y_fit = to_numpy(y_raw, dtype=None)
    splitter: Any
    if spec.estimator == "logistic":
        splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    else:
        splitter = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    losses = []
    for tr, te in splitter.split(X, y_fit if spec.estimator == "logistic" else None):
        fitted = clone(est).fit(X[tr], y_fit[tr])
        losses.append(_heldout_loss(spec, fitted, X[te], y_enc[te]))
    arr = np.asarray(losses, dtype=float)
    return float(np.std(arr, ddof=1) / np.sqrt(cv)), float(np.mean(arr))


def _resolve_tolerance(
    tolerance: Tolerance,
    est: Any,
    X: Array,
    y_raw: Any,
    y_enc: Array,
    spec: LinearModelSpec,
    cv: int,
    random_state: Optional[int],
) -> Tuple[float, str, str]:
    """Return (epsilon, epsilon_mode, human description)."""
    if isinstance(tolerance, str):
        key, alpha = tolerance.lower(), 0.05
        if key == "cv":
            se, mean_loss = cv_loss_standard_error(est, X, y_raw, y_enc, spec, cv, random_state)
            if not np.isfinite(se) or se <= 0.0:
                warnings.warn("Cross-validated loss has zero spread; falling back to tolerance=0.01.", stacklevel=3)
                return 0.01, "percent_loss", "1% of the optimal training loss (CV standard error was zero)"
            return (
                se,
                "absolute",
                f"one standard error of the {cv}-fold cross-validated {spec.loss_name}, "
                f"whose mean is {mean_loss:.4g}; models closer than this cannot be told apart by CV",
            )
        if key == "lr":
            return alpha, "LR_alpha", f"likelihood-ratio set at level alpha={alpha:g}"
        raise ValueError("tolerance string must be 'cv' or 'lr'")
    if isinstance(tolerance, tuple):
        if len(tolerance) != 2:
            raise ValueError("tuple tolerance must be ('lr', alpha) or ('absolute', gap)")
        kind, value = tolerance[0].lower(), float(tolerance[1])
        if kind == "lr":
            return value, "LR_alpha", f"likelihood-ratio set at level alpha={value:g}"
        if kind == "absolute":
            return value, "absolute", f"an explicit loss gap of {value:.4g}"
        raise ValueError("tuple tolerance must be ('lr', alpha) or ('absolute', gap)")
    r = float(tolerance)
    if not (0.0 < r < 1.0):
        raise ValueError("a float tolerance is a relative loss increase and must lie in (0, 1)")
    return r, "percent_loss", f"{r:.1%} of the optimal training loss"
