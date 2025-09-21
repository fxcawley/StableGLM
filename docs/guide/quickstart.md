# Quickstart

Minimal end-to-end usage of `RashomonSet`.

```python
import numpy as np
from rashomon import RashomonSet

rng = np.random.default_rng(0)
X = rng.normal(size=(100, 5))
w = rng.normal(size=5)
p = 1.0 / (1.0 + np.exp(-(X @ w)))
y = (rng.random(100) < p).astype(float)

rs = RashomonSet(estimator="logistic", random_state=0).fit(X, y)
print(rs.diagnostics())

# Probability bands for first 3 rows
print(rs.probability_bands(X[:3]))

# Coefficient intervals and ellipsoid samples
print(rs.coef_intervals())
print(rs.sample_ellipsoid(n_samples=5))
```

Notes:
- Install dev requirements to build docs and run tests: `pip install -r requirements-dev.txt`.
- Install the package in editable mode for local development: `pip install -e .`.
