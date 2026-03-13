# Quickstart

Install and run a complete Rashomon set analysis in under 20 lines.

## Install

```bash
pip install -e .
```

## Minimal Example

```python
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from rashomon import RashomonSet

# Load data
X, y = load_breast_cancer(return_X_y=True)
X = StandardScaler().fit_transform(X[:, :5])

# Fit Rashomon set (3% loss tolerance)
rs = RashomonSet(
    estimator="logistic",
    epsilon=0.03,
    epsilon_mode="percent_loss",
    random_state=0,
).fit(X, y.astype(float))

# How much do predictions vary across equally-good models?
amb = rs.ambiguity(X, threshold_mode="fixed", threshold_value=0.5)
print(f"Ambiguity: {amb['ambiguity_rate']:.1%} of patients get unstable diagnoses")

# How do coefficient ranges compare to bootstrap CIs?
comp = rs.compare_to_bootstrap(X, y.astype(float), n_bootstrap=200, n_rashomon=200)
for name, d in comp["divergence"].items():
    print(f"  {name}: Rashomon interval is {d['width_ratio']:.0f}x wider than bootstrap CI")
```

## What to Read Next

- {doc}`concepts <concepts>`: What are Rashomon sets, and why do they matter?
- {doc}`../examples/tutorial`: Full case study showing bootstrap CIs vs. Rashomon analysis
- {doc}`../api/reference`: Complete API documentation
