# Quickstart

## Install

```bash
pip install rashomon-py
```

## Audit a model you already have

```python
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from rashomon import audit

data = load_breast_cancer(as_frame=True)
X, y = data.data.iloc[:, :10], data.target

model = make_pipeline(StandardScaler(), LogisticRegression()).fit(X, y)

report = audit(model, X, y, random_state=0)
print(report.summary())
report.plot()
```

Three things to look at:

1. **Flip rate.** `report.flip_rate` is the share of rows for which some equally-good model predicts the other label. `report.flipped` is the row mask, so `X[report.flipped]` selects those rows.
2. **Sign stability.** `report.coefficients` lists each coefficient's range across all equally-good models. `sign_stable=False` means the direction of that effect is not determined by this data and model class.
3. **Reliability.** The header line reports the sampler's effective sample size. If it is not labelled "reliable", increase `n_samples`.

New rows go through the same pipeline:

```python
report.predict_ranges(X_new)      # estimate, low, high, flipped per row
```

## Change what "equally good" means

```python
audit(model, X, y, tolerance=0.01)             # models at most 1% worse than optimal
audit(model, X, y, tolerance="lr")             # likelihood-ratio set, alpha = 0.05
audit(model, X, y, tolerance=("absolute", 0.002))
```

The default `"cv"` uses one standard error of the cross-validated loss; see {doc}`choosing_epsilon`.

## Regression

`Ridge`, `RidgeCV` and `LinearRegression` work the same way. Regression has no label to flip, so pass a decision cutoff if you have one:

```python
from sklearn.linear_model import Ridge
report = audit(Ridge(alpha=1.0).fit(X, y), X, y, threshold=50_000)   # "approved if predicted income > 50k"
```

## Expert API

```python
from rashomon import RashomonSet
rs = RashomonSet.from_sklearn(model, X, y, epsilon=0.02, random_state=0)
rs.coef_extremes()                      # exact coefficient ranges (intercept first)
rs.functional_range(x_row)              # exact range of one row's logit
samples = rs.sample_hitandrun(2000, ellipsoid_mix=0.5)
```

## What to read next

- {doc}`when_to_use`: what this answers and what it does not
- {doc}`concepts`: the ε-Rashomon set and the quantities computed over it
- {doc}`../examples/tutorial`: a full case study
- {doc}`../api/reference`: API documentation
