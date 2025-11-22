# Core Concepts

## The Rashomon Effect

In machine learning, the "Rashomon Effect" describes the observation that for a given dataset, there are often many different models that perform approximately equally well. These models may have vastly different internal structures, feature importances, or predictions for individual instances.

**Rashomon-GLM** allows you to explore this set of models (the "Rashomon Set") for Generalized Linear Models (GLMs).

## The Rashomon Set

Formally, let $\mathcal{F}$ be the class of linear models (parameterized by $\theta$). Let $L(\theta)$ be the loss function. The empirical risk minimizer is $\hat{\theta} = \arg\min_\theta L(\theta)$.

The **$\epsilon$-Rashomon set** is defined as:

$$ R_\epsilon = \{ \theta \in \mathcal{F} : L(\theta) \le L(\hat{\theta}) + \epsilon \} $$

where $\epsilon$ is a user-defined tolerance. This set contains all models that are "good enough."

### Why explore it?

1.  **Robustness**: If feature $X$ is important in the optimal model but has zero weight in many other good models, its importance is unstable.
2.  **Fairness**: Two models might have the same accuracy but treat a specific subgroup differently. This is "predictive multiplicity."
3.  **Domain Constraints**: You might prefer a sparser model or one that aligns with causal knowledge, even if it has slightly higher loss.

## Predictive Multiplicity

Predictive multiplicity refers to the phenomenon where models in the Rashomon set assign conflicting predictions to the same input.

### Ambiguity
**Ambiguity** measures the fraction of samples for which the models in the Rashomon set disagree on the label. If a loan applicant is rejected by the optimal model but accepted by 40% of the models in the Rashomon set, their prediction is "ambiguous."

### Discrepancy
**Discrepancy** is the maximum disagreement rate between any two models in the set. It answers: "How different can two equally good models be?"

## Variable Importance

Standard feature importance (e.g., coefficient magnitude) gives a single number. **Variable Importance Clouds (VIC)** visualize the distribution of coefficients across the entire Rashomon set, giving you a full picture of feature stability.

**Model Class Reliance (MCR)** provides formal bounds on the importance of a variable (highest possible importance vs. lowest possible importance) across the set.

