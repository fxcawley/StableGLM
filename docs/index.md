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
:caption: Contents

guide/concepts
guide/quickstart
examples/tutorial
evaluation
guide/overview
api/reference
reproducibility
```
