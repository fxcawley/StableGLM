# Overview

## What StableGLM Does

StableGLM is a stability auditing tool for L2-regularized linear and logistic
regression. It characterizes the $\epsilon$-Rashomon set -- the set of all parameter
vectors achieving loss within $\epsilon$ of the optimum -- and reports whether
predictions, feature importances, and individual decisions are robust.

**Certificates** (closed-form, fast):
- **Hacking intervals**: the range of any linear functional $s^\top\theta$ across the Rashomon set
- **Coefficient intervals**: per-feature parameter ranges
- **Probability bands**: per-instance prediction ranges
- **Ambiguity**: fraction of instances where the Rashomon set contains models that disagree on the label (Marx, Calmon, Ustun 2020)

**Sampling** (exact, slower):
- **Ellipsoid sampling**: fast but approximate (quadratic approximation of the loss)
- **Hit-and-Run**: exact membership sampling from the true Rashomon set via MCMC

**Analysis**:
- **Coefficient distributions**: spread of each parameter across the Rashomon set
  (inspired by Dong & Rudin 2020, adapted to GLMs)
- **MCR** (Model Class Reliance): min/max feature importance bounds (Fisher, Rudin, Dominici 2019)
- **Discrepancy**: worst-case disagreement rate between any two models in the set (Marx et al. 2020)
- **Bootstrap/Bayesian comparison**: side-by-side comparison with classical uncertainty measures

## Why This Matters

Standard model evaluation reports a single fit and quantifies *sampling uncertainty*
(bootstrap CIs, p-values). Rashomon set analysis answers a different question: *"how
many qualitatively different models perform nearly as well?"*

For GLMs, the Rashomon set is closely related to likelihood ratio confidence regions
from classical statistics -- the `LR_alpha` calibration mode makes this bridge explicit.
The contribution is providing a unified, calibrated, sklearn-native interface that
connects classical statistical sensitivity analysis with modern predictive multiplicity
metrics.

The {doc}`evaluation <../evaluation>` shows that at d=10, certificate-based ambiguity
tracks empirical (Hit-and-Run) ambiguity within 1.3x, making the certificates a fast
and reliable screening tool.

## Supported Models

| Model | Loss | Regularization |
|-------|------|----------------|
| Logistic regression | $\frac{1}{n}\sum[\log(1+e^z) - yz] + \frac{\lambda}{2}\|\theta\|^2$ | L2 |
| Ridge regression | $\frac{1}{2n}\sum(y-z)^2 + \frac{\lambda}{2}\|\theta\|^2$ | L2 |

## Links

- {doc}`quickstart`: Get running in 20 lines
- {doc}`../examples/tutorial`: Full case study with real data
- {doc}`../evaluation`: Scaling, tightness, and certificate calibration
- {doc}`../api/reference`: API documentation
