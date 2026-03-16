# rashomon-py

**Stability auditing for GLMs in high-stakes sklearn workflows.**

Answers the question: *"Would my conclusions change if I used a different
equally-good model?"*

rashomon-py characterizes the epsilon-Rashomon set for L2-regularized logistic
and linear regression -- the set of all parameter vectors achieving loss within
epsilon of the optimum -- and audits whether predictions, feature importances,
and individual decisions are robust to the choice of model within that set.

Two computation modes: fast ellipsoidal certificates (closed-form, milliseconds)
and exact Hit-and-Run MCMC sampling (slower, provably correct). The certificates
track ground truth within 1.3x at d=10 and provide valid upper bounds at any
dimensionality.

## Install

```bash
pip install -e .
```

## Quickstart

```python
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from rashomon import RashomonSet

X, y = load_breast_cancer(return_X_y=True)
X = StandardScaler().fit_transform(X[:, :5])

rs = RashomonSet(estimator="logistic", epsilon=0.03,
                 epsilon_mode="percent_loss", random_state=0).fit(X, y.astype(float))

# How many patients get unstable diagnoses?
amb = rs.ambiguity(X, threshold_mode="fixed", threshold_value=0.5)
print(f"Ambiguity: {amb['ambiguity_rate']:.1%}")

# Coefficient stability across near-optimal models
vic = rs.variable_importance_cloud(n_samples=200)
print(f"Feature std: {vic['std']}")
```

## Documentation

- [Concepts](docs/guide/concepts.md) -- Rashomon sets, predictive multiplicity, three kinds of uncertainty
- [Tutorial](docs/examples/tutorial.md) -- Full case study on Breast Cancer
- [Evaluation](docs/evaluation.md) -- Scaling, tightness, certificate calibration on 4 real datasets
- [API Reference](docs/api/reference.rst) -- Complete API docs

## Key References

- Fisher, Rudin, Dominici (2019). "All Models are Wrong, but Many are Useful." *JMLR*. -- MCR framework
- Marx, Calmon, Ustun (2020). "Predictive Multiplicity in Classification." *ICML*. -- Ambiguity/discrepancy metrics
- Dong & Rudin (2020). "Exploring the Cloud of Variable Importance." *Nature Machine Intelligence*. -- VIC concept (tree models; adapted here for GLMs)
