# Overview

## What rashomon-py computes

rashomon-py characterizes the $\varepsilon$-Rashomon set for L2-regularized logistic and linear regression: the set of all parameter vectors achieving loss within $\varepsilon$ of the optimum. It then computes stability metrics over this set, answering the question of whether predictions, feature importances, and individual decisions are robust to the choice of model within the set.

The loss surface for a GLM with L2 regularization is convex, so the Rashomon set is a convex sublevel set. Near the optimum, the Hessian $H = \nabla^2 L(\hat\theta)$ provides a local ellipsoidal approximation $\mathcal{E}_\varepsilon = \{\hat\theta + \Delta : \Delta^\top H \Delta \leq 2\varepsilon\}$ that is analytically tractable. For exact computations over the true (non-ellipsoidal) set, the toolkit uses hit-and-run sampling with a membership oracle.

**Certificates** (closed-form, fast):
- Coefficient intervals: per-feature parameter ranges across $\mathcal{E}_\varepsilon$
- Probability bands: per-instance prediction ranges
- Ambiguity upper bounds: fraction of instances where the set contains models that disagree on the label (Marx, Calmon, & Ustun, 2020)

**Sampling** (asymptotically exact, slower):
- Ellipsoid sampling: fast but approximate (based on the quadratic approximation)
- Hit-and-run: asymptotically exact membership sampling from the true Rashomon set via MCMC

**Analysis**:
- Coefficient distributions across the Rashomon set, inspired by the Variable Importance Cloud of Dong & Rudin (2020), adapted to the GLM setting
- Model Class Reliance: min/max feature importance bounds (Fisher, Rudin, & Dominici, 2019)
- Discrepancy: worst-case disagreement rate between any two models in the set (Marx et al., 2020)
- Bootstrap and Bayesian comparison: side-by-side comparison with classical uncertainty measures

## Why this matters

Standard model evaluation reports a single fit and quantifies sampling uncertainty (bootstrap CIs, p-values). The Rashomon set addresses a different question: how many qualitatively different models perform nearly as well? If the answer is "many," then any single model's explanations are, to some degree, artifacts of the particular optimum the solver found. Fisher, Rudin, & Dominici (2019) make this point in the context of variable importance; Marx, Calmon, & Ustun (2020) formalize it for predictions.

For GLMs, the Rashomon set is closely related to likelihood-ratio confidence regions from classical statistics. The `LR_alpha` calibration mode makes this bridge explicit, setting $\varepsilon = \chi^2_{d,1-\alpha} / (2n)$ via Wilks' theorem.

## Supported models

| Model | Loss | Regularization |
|-------|------|----------------|
| Logistic regression | $\frac{1}{n}\sum[\log(1+e^z) - yz] + \frac{\lambda}{2}\lVert\theta\rVert^2$ | L2 |
| Ridge regression | $\frac{1}{2n}\sum(y-z)^2 + \frac{\lambda}{2}\lVert\theta\rVert^2$ | L2 |

## References

- Fisher, A., Rudin, C., & Dominici, F. (2019). All models are wrong, but many are useful. *JMLR*, 20(177), 1--81.
- Marx, C., Calmon, F., & Ustun, B. (2020). Predictive multiplicity in classification. *ICML*.
- Dong, J. & Rudin, C. (2020). Exploring the cloud of variable importance for the set of all good models. *Nature Machine Intelligence*, 2, 810--824.
