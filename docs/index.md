# rashomon-py

Stability auditing for GLMs.

Answers the question: *"Would my conclusions change if I used a different
equally-good model?"* -- in two lines of code.

rashomon-py characterizes the set of all L2-regularized linear and logistic
regression models that perform within $\epsilon$ of the optimum, then audits
whether predictions, feature importances, and individual decisions are
robust to the choice of model within that set.

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
