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

`RashomonSet` supports four calibration modes (`audit()` exposes them through its `tolerance` argument; see {doc}`choosing_epsilon`):
- `percent_loss`: $\epsilon = \rho \cdot L(\hat{\theta})$ for a user-specified $\rho$
- `LR_alpha`: $\epsilon = \chi^2_{d,1-\alpha} / (2n)$ via Wilks' theorem
- `absolute`: $\epsilon$ given directly, in the units of the loss (used by the `"cv"` and `("profile", alpha)` tolerances)

## Relation to classical intervals

For an L2-penalized GLM, $nL(\theta)$ is a negative log-posterior (log-likelihood plus a
Gaussian prior with precision $n\lambda$ on the penalized coordinates), so $\hat\theta$ is
its mode and the Laplace posterior is $N(\hat\theta, (nH)^{-1})$. To second order the
$\varepsilon$-Rashomon set is the highest-posterior-density region of that posterior with
mass $F_{\chi^2_d}(2n\varepsilon)$, and the coefficient ranges are
$\hat\theta_j \pm \sqrt{2n\varepsilon}\,\mathrm{sd}_j$. With `("profile", alpha)` and no
penalty they are the $(1-\alpha)$ profile-likelihood confidence intervals. The bootstrap
of the penalized estimator has the sandwich variance $(nH)^{-1}J(nH)^{-1}$, which equals
$(nH)^{-1}$ when $\lambda = 0$ and is smaller along the directions the penalty controls.
{doc}`why_not_bootstrap` states these relations with numbers; the
{doc}`case study <../examples/tutorial>` checks them on the breast-cancer model.

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

The min and max permutation importance of each feature over sampled models in the
Rashomon set (Fisher, Rudin, Dominici 2019); the sampled extremes are inner bounds on the
true ones. If the sampled MCR- < 0 for a feature, a near-optimal model exists where that
feature is actively harmful. A positive sampled MCR- does not establish that the feature
is indispensable across all good models, only across the models visited.
