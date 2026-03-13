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

### What $\epsilon$ controls

- **Small $\epsilon$**: tight set, only models very close to the optimum. Results are
  conservative -- if instability appears here, it is severe.
- **Large $\epsilon$**: permissive set, includes models with meaningfully higher loss.
  Results may overstate instability.

StableGLM supports three calibration modes:
- `percent_loss`: $\epsilon = \rho \cdot L(\hat{\theta})$ for a user-specified $\rho$
- `LR_alpha`: $\epsilon = \chi^2_{d,1-\alpha} / (2n)$ via Wilks' theorem
- `LR_alpha_highdim`: high-dimensional correction (experimental)

## Two Kinds of Uncertainty

Standard statistical tools (bootstrap CIs, Bayesian posteriors, p-values) quantify
**sampling uncertainty**: how much would the answer change if we drew a different dataset
from the same distribution?

Rashomon set analysis quantifies **design-choice multiplicity**: how many qualitatively
different models achieve nearly the same loss on *this* dataset?

These are orthogonal:

| | Narrow bootstrap CIs | Wide bootstrap CIs |
|---|---|---|
| **Narrow VIC** | Stable: one clear best model | Uncertain but robust predictions |
| **Wide VIC** | **Dangerous**: false confidence | Everything is unstable |

The "narrow CIs + wide VIC" case is the most concerning because standard tools report
confidence while the underlying predictions are arbitrary. The
{doc}`case study <../examples/tutorial>` demonstrates this with a 4-11x width ratio.

## Predictive Multiplicity

### Ambiguity

The fraction of instances where the Rashomon set contains models that disagree on the label.
Instance $i$ is **ambiguous** if its margin interval $[m_i^{\min}, m_i^{\max}]$ straddles
the decision threshold. These are people whose prediction is an artifact of model selection.

### Discrepancy

The maximum disagreement rate between any two models in the Rashomon set. If discrepancy
is 30%, there exist two equally-good models that give opposite predictions for 30% of
the population.

## Variable Importance

### VIC (Variable Importance Cloud)

The distribution of each coefficient across the Rashomon set. Unlike a confidence interval,
VIC does not shrink with more data -- it reflects the geometry of the loss surface, not
sampling noise. A feature with a wide VIC is one where many different weightings are
compatible with near-optimal loss.

### MCR (Model Class Reliance)

The min and max permutation importance of each feature across the Rashomon set
(Fisher, Rudin, Dominici 2019). If MCR- < 0 for a feature, there exists a near-optimal
model where that feature is not just unnecessary but actively harmful. If MCR- > 0,
the feature is indispensable across all good models.
