# StableGLM / rashomon-py

Rashomon-GLM: ε-Rashomon sets, certificates, exact membership sampling, and set-level interpretability metrics for GLMs.

- Proposal and math: see `prop.md`
- Project plan: see `PROJECT_PLAN.md`
- Reproducibility: see `REPRODUCIBILITY.md`

## Install (dev)

```bash
pip install -e .
pip install -r requirements-dev.txt
```

## Quickstart (skeleton)

```python
from rashomon import RashomonSet
rs = RashomonSet(random_state=0).fit(X, y)
print(rs.diagnostics())
```

Docs scaffold lives in `docs/`. CI files in `.github/workflows/`.
