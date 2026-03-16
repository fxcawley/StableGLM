# Changelog

All notable changes to rashomon-py will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-03-16

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

[0.1.0]: https://github.com/fxcawley/rashomon-py/releases/tag/v0.1.0
