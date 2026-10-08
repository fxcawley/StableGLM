# Reproducibility

## Environment

Use Python 3.10 or newer and install the project with its development dependencies:

```bash
python -m pip install -U pip
python -m pip install -e .
python -m pip install -r requirements-dev.txt
```

The core runtime dependencies are numpy, scipy, scikit-learn, and matplotlib. The documentation build additionally uses Sphinx, MyST, and the Read the Docs theme.

## Determinism

Set `random_state` on `RashomonSet` for deterministic sampling and diagnostics where supported:

```python
from rashomon import RashomonSet

rs = RashomonSet(estimator="logistic", epsilon=0.03, random_state=0).fit(X, y)
diag = rs.diagnostics()
```

Numerical results may vary slightly across BLAS implementations and platform libraries. Interpret sampled estimates through the reported effective sample size rather than expecting bitwise equality; the exact quantities (coefficient ranges, flips) are deterministic up to solver tolerance.

## Data

Large raw datasets are not committed to the repository. The Adult Census integration test skips when `tests/data/adult.data` is absent. To reproduce the Adult experiments locally, run:

```bash
python scripts/download_data.py
```

The evaluation script (`python scripts/evaluate.py`) fetches and caches the German Credit data from OpenML on first use and skips the Adult rows when `adult.data` is absent. It writes `docs/_static/evaluation.json` and prints the tables in the evaluation page.

## Verification

Run the test suite and documentation build:

```bash
python -m pytest tests
python -m sphinx -W -b html docs docs/_build/html
```
