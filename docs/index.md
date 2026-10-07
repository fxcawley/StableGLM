# rashomon-py

**Does your conclusion survive every equally-good model?**

Many parameter vectors fit a given dataset essentially as well as the one your solver returned (Breiman's "Rashomon effect"). rashomon-py audits a fitted scikit-learn logistic or linear regression for this *model multiplicity*: which predictions would flip, how far each coefficient can move, and which signs every equally-good model agrees on.

```python
from rashomon import audit
report = audit(model, X, y)      # a fitted LogisticRegression / Ridge / LinearRegression or Pipeline
print(report.summary())
report.plot()
```

Under the hood, the $\varepsilon$-Rashomon set is $\mathcal{R}_\varepsilon = \{\theta : L(\theta) \leq L(\hat\theta) + \varepsilon\}$. For convex losses with L2 regularization it is a convex sublevel set; the toolkit computes exact ranges of coefficients by convex optimization, samples the set exactly with hit-and-run (accelerated by ellipsoid proposals), and offers a Hessian-ellipsoid approximation for fast screening.

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
