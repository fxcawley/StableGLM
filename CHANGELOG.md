# Changelog

All notable changes to rashomon-py will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.1] - Unreleased

### Added

- Exact per-row flip test. `RashomonSet.min_loss_on_hyperplane(s, c)` solves
  `min L(theta) s.t. s^T theta = c` (closed form for linear models, Newton on the
  hyperplane for logistic), and `RashomonSet.can_flip(X, tau)` uses it to decide
  exactly whether any model in the set puts a row on the other side of the threshold.
  `audit(..., exact_flips="auto")` settles every row not already flipped by a sampled
  model this way whenever `n_undecided * n * d^2 <= 1e9`; the report labels the flip
  rate as exact or sampled (`flip_method`). On the README example the exact flip rate
  is 17.4% where 2,000 samples found 14.4%.
- `tolerance=("profile", alpha)`: epsilon = chi2_1(1-alpha)/(2n), under which the
  coefficient ranges of an unpenalized fit are the (1-alpha) profile-likelihood
  confidence intervals.
- Validation tests (`tests/test_validation.py`). External: on the UCLA
  graduate-admissions logit example, `coef_extremes()` reproduces all twelve bounds of
  R's `confint()` to 1e-4 and the MLE to 5e-7, and agrees with an independent profile
  root-finder to 2e-6. Internal: a gridded two-dimensional Rashomon set checks the
  exact ranges, the exact flip test and the sampler's moments against brute force.
- `model_class_reliance(..., metric="loss_ratio")`: the Fisher-Rudin-Dominici reliance
  `loss(permuted)/loss(original)` with log-loss or MSE, alongside the default score
  drop.

### Changed

- Effective sample size uses Geyer's initial positive sequence estimator instead of
  the lag-1 AR(1) approximation. It accounts for autocorrelation at all lags and is
  more conservative (218 vs 387 on the README example).
- `model_class_reliance` no longer permutes the intercept column, and its docstring
  states that `mcr_min` / `mcr_max` are extremes over the sampled models.

### Fixed

- `compute_threshold(mode="match_prevalence")` returned `logit(prevalence)`, which
  does not make the predicted positive rate match the prevalence. It now returns the
  `(1 - prevalence)` quantile of the margins (training margins, or those of `X_val`).

## [0.3.0] - 2026-10-07

### Added

- `class_weight` and `sample_weight`. `audit(model, X, y, sample_weight=w)` and
  `RashomonSet.from_sklearn(..., sample_weight=w)` reconstruct scikit-learn's weighted
  objective: each row's weight is `sample_weight` times its class weight, the data term
  is the weighted mean loss, and the regularization strength is converted relative to
  the sum of weights (`lambda = 1/(C * sum(w))` for logistic, `alpha / sum(w)` for ridge).
  `RashomonSet.fit(..., sample_weight=w)` exposes the same in the lower-level API; the
  oracle, Hessian, samplers, exact ranges and bootstrap calibration all use the weighted
  objective. The `"cv"` tolerance refits with the weights and averages held-out losses
  with them. Row counts in the report (flip rate, disagreement) stay unweighted.

### Changed

- Models with `class_weight` no longer raise in `audit()`.

## [0.2.0] - 2026-10-07

The package is now built around one question, asked of a fitted scikit-learn
model: does your conclusion survive every equally-good model?

### Added

- `rashomon.audit(model, X, y)`: one-call audit of a fitted `LogisticRegression` /
  `LogisticRegressionCV` (binary, L2 or unpenalized), `Ridge` / `RidgeCV`,
  `LinearRegression`, or a `Pipeline` ending in one of these. Accepts pandas input and
  infers feature names. Returns a `StabilityReport` with `flip_rate`, `flipped`,
  `max_disagreement`, `coefficients` (with `sign_stable`), `prediction_ranges`,
  `predict_ranges(X_new)`, `summary()` and `plot()`.
- Default tolerance `"cv"`: one standard error of the cross-validated loss (the
  one-standard-error rule). Also `float` (relative loss), `"lr"` / `("lr", alpha)` and
  `("absolute", gap)`.
- `RashomonSet.from_sklearn(model, X, y, ...)` with exact conversion of the
  regularization strength and the unpenalized intercept; the audit verifies that the
  fitted coefficients sit at the reconstructed optimum and reports any mismatch.
- Exact ranges over the true Rashomon set: `RashomonSet.functional_range(s)` and
  `coef_extremes()` solve the convex programs max/min sᵀθ s.t. L(θ) ≤ L(θ̂) + ε
  (closed form for linear models; Lagrangian boundary tracing with warm-started
  Newton solves for logistic). `audit()` uses them for coefficient ranges and sign
  stability whenever `n · d² ≤ 5e6`.
- `sample_hitandrun(..., ellipsoid_mix=p)`: independence proposals from the Hessian
  ellipsoid, accepted with the Metropolis-Hastings rule for a uniform target. Raises
  the effective sample size per step by about a factor of d in low to moderate
  dimension; the chain still samples the exact set.
- `RashomonSet(penalize_intercept=...)` (default `False`, the scikit-learn
  convention), `C=np.inf` for unpenalized fits (λ = 0), `epsilon_mode="absolute"`,
  and `fit(..., theta_init=...)` warm starts.
- Damped-Newton polishing of the fitted optimum, so `theta_hat` is the minimizer of
  the stated objective rather than an L-BFGS iterate stopped at solver tolerance.

### Fixed

- Ellipsoid sampler orientation. `sample_ellipsoid` mapped ball samples with `L⁻¹`
  instead of `L⁻ᵀ` (where `H = LLᵀ`), which gives an ellipsoid with the right size but
  the wrong orientation; only about 30% of the draws lay in the Hessian ellipsoid or in
  the Rashomon set. Every quantity in 0.1.0 computed from ellipsoid samples (VIC with
  the default sampler, MCR, the quoted set-fidelity figures) was affected. Hit-and-run
  samples and the closed-form ellipsoid intervals were not.
- The near-separation guard no longer rejects penalized fits with a few confidently
  classified points. It applies only to unpenalized fits, where separation means the
  optimum does not exist.
- Refitting a `RashomonSet` now resets the cached Hessian, Cholesky factor and
  preconditioner.
- `get_params()` now includes `fit_intercept`.
- The `C` docstring claimed scikit-learn semantics. `RashomonSet.C` is `1/λ` for the
  mean-loss objective, i.e. scikit-learn's `C` times `n`. The docstring now says so,
  and `audit` / `from_sklearn` do the conversion.
- Logistic fits now validate that `y` is in {0, 1}.

### Changed

- With `fit_intercept=True` the intercept is no longer L2-penalized by default
  (`penalize_intercept=False`), matching scikit-learn.
- `pandas` is a new dependency. Project URLs point at the fxcawley/StableGLM repository.

## [0.1.0] - 2026-03-16

First public release.

### Added

- `RashomonSet` class for L2-regularized logistic and linear regression.
- Certificate-based (ellipsoidal) computation: coefficient intervals,
  probability bands, ambiguity upper bounds. Closed-form, milliseconds.
- Hit-and-Run MCMC sampling from the true Rashomon set with ESS diagnostics.
- Ambiguity metric: fraction of instances with unstable predictions
  (Marx, Calmon, Ustun 2020).
- Discrepancy metric: worst-case pairwise disagreement rate.
- Variable Importance Cloud (VIC): coefficient distributions across the
  Rashomon set (inspired by Dong & Rudin 2020, adapted to GLMs).
- Model Class Reliance (MCR): min/max permutation importance bounds
  (Fisher, Rudin, Dominici 2019).
- Bootstrap and Bayesian comparison methods.
- Three epsilon calibration modes: `percent_loss`, `LR_alpha`,
  `LR_alpha_highdim`.
- Plotting helpers: `plot_vic`, `plot_ambiguity`, `plot_discrepancy`.
- sklearn-compatible API: `fit`, `predict`, `score`, `get_params`, `set_params`.
- Evaluation on 4 real datasets (Breast Cancer, German Credit, Adult Census).
- Workflow documentation: choosing epsilon, certificates vs sampling,
  interpreting instability, bootstrap comparison.
- Canonical end-to-end notebook (`examples/stability_audit.ipynb`).

### Scope

L2-regularized logistic and linear regression only. No trees, neural nets,
L1 penalties, or arbitrary estimators.

[0.3.1]: https://github.com/fxcawley/StableGLM/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/fxcawley/StableGLM/releases/tag/v0.3.0
[0.2.0]: https://github.com/fxcawley/StableGLM/releases/tag/v0.2.0
[0.1.0]: https://github.com/fxcawley/StableGLM/releases/tag/v0.1.0
