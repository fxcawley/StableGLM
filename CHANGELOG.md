# Changelog

All notable changes to rashomon-py will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-10-07

Repositioned around a single question -- *does your conclusion survive every
equally-good model?* -- asked of the scikit-learn model you already have.

### Added

- `rashomon.audit(model, X, y)`: one-call audit of a fitted `LogisticRegression` /
  `LogisticRegressionCV` (binary, L2 or unpenalized), `Ridge` / `RidgeCV`,
  `LinearRegression`, or a `Pipeline` ending in one of these. Accepts pandas input and
  infers feature names. Returns a `StabilityReport` with plain-language fields
  (`flip_rate`, `flipped`, `max_disagreement`, `coefficients` with `sign_stable`,
  `prediction_ranges`, `predict_ranges(X_new)`), `summary()` and `plot()`.
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
  ellipsoid, accepted with the exact Metropolis–Hastings rule for a uniform target.
  Raises the effective sample size per step by roughly a factor of d in low to
  moderate dimension while still sampling the exact set.
- `RashomonSet(penalize_intercept=...)` (default `False`, the scikit-learn
  convention), `C=np.inf` for unpenalized fits (λ = 0), `epsilon_mode="absolute"`,
  and `fit(..., theta_init=...)` warm starts.
- Damped-Newton polishing of the fitted optimum, so `theta_hat` is the true minimizer
  of the stated objective rather than an L-BFGS iterate stopped at solver tolerance.

### Fixed

- **Ellipsoid sampler orientation.** `sample_ellipsoid` mapped ball samples with
  `L⁻¹` instead of `L⁻ᵀ` (where `H = LLᵀ`), producing samples from an ellipsoid with
  the right size but the wrong orientation; only ~30% of "ellipsoid" draws lay in the
  Hessian ellipsoid or in the Rashomon set. Every quantity in 0.1.0 computed from
  ellipsoid samples (VIC with the default sampler, MCR, the quoted "set fidelity"
  figures) was affected. Hit-and-run samples and the closed-form ellipsoid intervals
  were not.
- The near-separation guard no longer rejects ordinary penalized fits with a few
  confidently classified points; it applies only to unpenalized fits, where
  separation means the optimum does not exist.
- Refitting a `RashomonSet` now resets the cached Hessian, Cholesky factor and
  preconditioner.
- `get_params()` now includes `fit_intercept`.
- The `C` docstring claimed scikit-learn semantics; `RashomonSet.C` is `1/λ` for the
  mean-loss objective, i.e. scikit-learn's `C` times `n`. Documented, and avoided
  entirely by `audit` / `from_sklearn`.
- Logistic fits now validate that `y` is in {0, 1}.

### Changed

- With `fit_intercept=True` the intercept is no longer L2-penalized by default
  (`penalize_intercept=False`), matching scikit-learn.
- `pandas` is a new dependency; project URLs point at the actual repository.

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

[0.2.0]: https://github.com/fxcawley/StableGLM/releases/tag/v0.2.0
[0.1.0]: https://github.com/fxcawley/StableGLM/releases/tag/v0.1.0
