# Reproducibility Guide

This document specifies the environment, seeding, commands, and artifact policy to reproduce results and figures for StableGLM.

## Environment

- Python: 3.9–3.12 (tested)
- OS: Linux/macOS/Windows (CI matrix)
- BLAS: OpenBLAS and MKL tested; BLAS vendor logged in diagnostics
- Core deps (minimal):
  - numpy, scipy, scikit-learn
  - matplotlib, seaborn
  - tqdm
  - Optional (CPU only): torch (for HVP/Lanczos conveniences)
  - docs: sphinx, myst-parser
  - testing: pytest, pytest-cov

We will publish an `environment.yml` and `requirements.txt` upon first release. For now, a minimal dev install:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements-dev.txt  # provided later
```

## Seeding & Determinism

- Global seed is set via `random_state` parameter on `RashomonSet` and propagated to numpy/torch RNGs.
- We log: seed, numpy version, scipy version, sklearn version, torch version (if used), and BLAS vendor in `rs.diagnostics()`.
- Tests assert numerical tolerance across BLAS (not bitwise equality).

Example:

```python
from rashomon import RashomonSet
rs = RashomonSet(random_state=0).fit(X, y)
diag = rs.diagnostics()
print(diag["seed"], diag["blas_vendor"])  # example fields
```

## Running examples (placeholder)

Until the package is published, examples will live under `examples/`.

- Ellipsoid certificates on a toy dataset:
```bash
python examples/ellipsoid_demo.py --epsilon 0.02 --epsilon-mode percent_loss --measure lr --seed 0
```

- Hit-and-Run sampling diagnostics:
```bash
python examples/sampler_diag.py --epsilon 0.02 --steps 2000 --burnin 200 --thin 5 --seed 0
```

- Metrics demo (VIC/MCR/Ambiguity):
```bash
python examples/metrics_demo.py --epsilon 0.02 --threshold fixed --perm iid --seed 0
```

We will wire these into CI smoke tests to ensure they run < 3 minutes CPU.

## Evaluation protocol

- Tightness study: compare analytic bounds with empirical extrema over an ε-grid; record gap statistics and wall clock.
- Scaling study: n ∈ {1e3, 1e4, 1e5}, d ∈ {10, 100, 1000}; report HVP time, CG iters, ESS/min.
- Fairness metrics: report threshold regimes (T1 fixed, T2 match_prevalence or match_fpr, T3 cost_opt). By default, report T1 & T2.

All evaluation scripts accept `--seed`, `--n-rep`, and write JSON/CSV artifacts.

## Artifact policy

- Do not commit large raw data; use `data/` (gitignored). Provide scripts to download/preprocess public datasets.
- Commit configuration files and small JSON summaries under `artifacts/` (limit < 5 MB per file).
- Figures are regenerated; do not store large PNGs unless necessary. For paper figures, store vector-friendly versions (SVG/PDF) if small.
- Every experiment run writes a `run.json` with: seed, software versions, BLAS vendor, commit hash, start/end time, and key diagnostics (κ(H), ESS/min, set-fidelity).

## Git integration

- Initialize git in the project root, commit docs, and configure a remote (e.g., GitHub). CI will pick up environment checks and smoke examples.

Commands:

```bash
# one-time setup
git init
git add .
git commit -m "chore: init reproducibility docs and project plan"
# add remote (replace URL)
git remote add origin https://github.com/ORG/StableGLM.git
git branch -M main
git push -u origin main
```

## CI hooks

- Matrix on py3.9–3.12 and OpenBLAS/MKL runners.
- Jobs: unit tests, smoke examples (<3 min), docs build, wheel build.
- Artifacts: upload coverage.xml, example JSON summaries.

## Contact

Please open an issue with the label `[repro]` for any reproducibility gap.
