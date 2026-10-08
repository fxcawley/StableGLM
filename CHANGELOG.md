# Changelog

All notable changes to rashomon-py will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.2] - Unreleased

Robustness pass over the existing scope before widening it.

### Added

- `scripts/evaluate.py` and a rewritten evaluation page comparing exact, sampled and
  ellipsoid quantities on four real datasets (d = 10 to 101). Sampled coefficient
  ranges cover 75% of the exact range at d = 10 and 20% at d = 101; sampled flip rates
  are a third to a quarter of the exact ones on the larger sets; the Hessian ellipsoid
  is within a few percent of exact at tolerances of 1-3% of the loss.
- Sparse `X` (scipy.sparse) is densified up to 20 million entries; larger inputs get an
  explanatory error.
- `audit()` and `from_sklearn` raise if `X` has column names that differ from the ones
  the model was fitted on (a reordered DataFrame would otherwise audit the wrong model).
- `RashomonSet.can_flip(..., coef_box=...)` screens rows with the exact coefficient box
  before optimising; `min_loss_on_hyperplane(..., theta0=...)` accepts a warm start.
- The report header says `(weighted)` when weights are in use, and a note reports an
  ill-conditioned Hessian.
- The example notebook is rewritten around `audit()` and executed by the test suite.

### Changed

- The exact flip test starts from the quadratic model's minimiser on the hyperplane and
  settles rows already inside the set with one vectorised loss evaluation; 4-5x faster
  on large problems (about 45 s at n = 5,000, d = 101). The automatic gate is now
  `n_undecided * n * d^2 <= 2e10`, and the note for larger problems estimates the cost
  of `exact_flips=True`.
- Hessian conditioning: condition numbers between 1e8 and 1e12 warn instead of raising
  (unstandardized features typically land here), above 1e12 the error explains the
  cause (collinear or constant features without a penalty, or wildly different scales)
  and the remedy.
- Membership tests use a floating-point slack of `min(tol, 1e-3 * epsilon)` instead of
  the absolute `tol`, which inflated very small sets.
- `_sigmoid` no longer evaluates `exp` of large positive arguments (spurious overflow
  warnings).
- Hit-and-run with `ellipsoid_mix > 0` falls back to plain hit-and-run when the Hessian
  is not numerically SPD, and the step-cap error explains that the set is too small to
  resolve.
- `python_requires` is now `>= 3.10`; 3.9 was never tested in CI and is end-of-life.
- `scripts/benchmark_scale.py` passed scikit-learn's `C` straight to `RashomonSet`
  (which expects `C * n`); fixed, and the script is marked as superseded.

### Fixed

- `variable_importance_cloud`, `shapley_vic` and `compare_to_bootstrap` accept feature
  names of the features alone (the intercept label is prepended) as well as of all
  coordinates. `compare_to_bootstrap` labelled feature-level results with the
  coordinate names, which were off by one when an intercept was fitted.
- The example notebook called attributes that did not exist (`epsilon_`, `vic["min"]`,
  `comp["comparison"]`).

## [0.3.1] - 2026-10-07

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

[0.3.2]: https://github.com/fxcawley/StableGLM/compare/v0.3.1...HEAD
[0.3.1]: https://github.com/fxcawley/StableGLM/releases/tag/v0.3.1
[0.3.0]: https://github.com/fxcawley/StableGLM/releases/tag/v0.3.0
[0.2.0]: https://github.com/fxcawley/StableGLM/releases/tag/v0.2.0
[0.1.0]: https://github.com/fxcawley/StableGLM/releases/tag/v0.1.0
