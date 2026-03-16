# Core Concepts

## The Rashomon Effect

For a given dataset, there are often many models that perform approximately equally well.
These models may have different coefficients, different feature importances, and different
predictions for individual instances. The name comes from Breiman (2001), who observed that
the multiplicity of good models is a fundamental property of many prediction problems.

## The Rashomon Set

Given a loss function $L(\theta)$ with minimizer $\hat{\theta}$, the **$\epsilon$-Rashomon set** is:

$$ R_\epsilon = \{ \theta : L(\theta) \le L(\hat{\theta}) + \epsilon \} $$

This is the set of all parameter vectors within $\epsilon$ of the best achievable loss.
For convex losses with L2 regularization (logistic regression, ridge regression), this set
is a convex sublevel set of the loss surface. Near the optimum, it is approximately
ellipsoidal, with shape determined by the Hessian $H = \nabla^2 L(\hat{\theta})$.

For GLMs, this is closely related to **likelihood ratio confidence regions**
from classical statistics. The `LR_alpha` calibration mode makes this connection
explicit: $\epsilon = \chi^2_{d,1-\alpha} / (2n)$ via Wilks' theorem.

### What $\epsilon$ controls

- **Small $\epsilon$**: tight set, only models very close to the optimum. Results are
  conservative -- if instability appears here, it is severe.
- **Large $\epsilon$**: permissive set, includes models with meaningfully higher loss.
  Results may overstate instability.

rashomon-py supports three calibration modes:
- `percent_loss`: $\epsilon = \rho \cdot L(\hat{\theta})$ for a user-specified $\rho$
- `LR_alpha`: $\epsilon = \chi^2_{d,1-\alpha} / (2n)$ via Wilks' theorem
- `LR_alpha_highdim`: high-dimensional correction (experimental)

## Three Kinds of Uncertainty

Standard statistical tools quantify different aspects of model uncertainty:

**Bootstrap CIs** measure **sampling uncertainty**: how much would $\hat\theta$ change
if we drew a different dataset? Width shrinks as $O(1/\sqrt{n})$.

**Bayesian posteriors** measure **epistemic uncertainty**: what is our belief distribution
over $\theta$ given the data and prior? Under the Laplace approximation with a Gaussian
prior, the posterior is an ellipsoid of the same shape as the Rashomon set but typically
wider (it integrates over all plausible parameter values, not just near-optimal ones).

**Rashomon coefficient intervals** measure **design-choice multiplicity**: how many
qualitatively different models achieve loss within $\epsilon$ of the optimum? Width
depends on $\epsilon$ and the loss surface geometry, not on $n$ directly.

The typical ordering is:

$$\text{Bootstrap} \ll \text{Rashomon} \ll \text{Bayesian}$$

| | Narrower Rashomon than Bayesian | Wider Rashomon than Bayesian |
|---|---|---|
| **Narrow bootstrap CIs** | Normal: standard tools suffice | Loss surface is pathologically flat |
| **Wide bootstrap CIs** | All three agree: everything is uncertain | Rashomon set is enormous |

The {doc}`case study <../examples/tutorial>` demonstrates this ordering with a 1.3x
certificate-to-empirical ratio at d=10.

## Predictive Multiplicity

### Ambiguity

The fraction of instances where the Rashomon set contains models that disagree on the label
(Marx, Calmon, Ustun 2020). Instance $i$ is **ambiguous** if its margin interval
$[m_i^{\min}, m_i^{\max}]$ straddles the decision threshold. These are people whose
prediction is an artifact of model selection.

### Discrepancy

The maximum disagreement rate between any two models in the Rashomon set
(Marx, Calmon, Ustun 2020). If discrepancy is 10%, there exist two equally-good models
that give opposite predictions for 10% of the population.

## Feature Stability

### Coefficient Distributions

The distribution of each coefficient across the Rashomon set, computed by sampling
parameter vectors and reporting their statistics. Inspired by the Variable Importance
Cloud of Dong & Rudin (2020), adapted to the GLM setting where raw coefficients serve
as the natural importance measure. (Dong & Rudin's original VIC uses SHAP-based
importance over tree models, which is a different object.)

A feature with a wide coefficient distribution is one where many different weightings
are compatible with near-optimal loss. This does not shrink with more data -- it
reflects the geometry of the loss surface.

### MCR (Model Class Reliance)

The min and max permutation importance of each feature across the Rashomon set
(Fisher, Rudin, Dominici 2019). If MCR- < 0 for a feature, there exists a near-optimal
model where that feature is not just unnecessary but actively harmful. If MCR- > 0,
the feature is indispensable across all good models.
