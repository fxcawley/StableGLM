# rashomon-py

For many practical problems, many parameter vectors achieve nearly the same loss. This is the Rashomon effect (Breiman, 2001). rashomon-py characterizes this set for L2-regularized logistic and linear regression and asks which predictions, feature importances, and individual decisions are stable across it.

The $\varepsilon$-Rashomon set is $\mathcal{R}_\varepsilon = \{\theta : L(\theta) \leq L(\hat\theta) + \varepsilon\}$, the set of all parameter vectors within $\varepsilon$ of optimal loss. For convex losses with L2 regularization, this is a convex sublevel set with an analytically tractable ellipsoidal approximation. The toolkit computes stability metrics over this set using either fast closed-form certificates or hit-and-run MCMC sampling from the true set.

```{toctree}
:maxdepth: 2
:caption: Getting Started

guide/quickstart
guide/when_to_use
guide/choosing_epsilon
```

```{toctree}
:maxdepth: 2
:caption: User Guide

guide/certificates_vs_sampling
guide/interpreting_instability
guide/why_not_bootstrap
examples/tutorial
```

```{toctree}
:maxdepth: 2
:caption: Reference

evaluation
guide/concepts
guide/overview
api/reference
reproducibility
```
