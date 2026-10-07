# rashomon-py — notes for agents and contributors

## What this is

A multiplicity audit for fitted scikit-learn linear models. Public API: `rashomon.audit(model, X, y)`
→ `StabilityReport` (plain-language front door) and `rashomon.RashomonSet` (expert API).
`rashomon/_sklearn.py` converts sklearn objectives (`lambda = 1/(C·n)` for logistic, `alpha/n` for ridge,
unpenalized intercept) into the mean-loss objective `RashomonSet` uses. `RashomonSet.C` is *not* sklearn's `C`.

## Verify before finishing

```bash
python -m pytest tests/ -q                       # ~30 s, 95+ tests
python -m ruff check rashomon tests scripts       # pinned in requirements-dev.txt
python -m mypy rashomon --ignore-missing-imports
python -m sphinx -W -b html docs docs/_build/html # CI builds docs with -W
python scripts/verify_tutorial_claims.py          # CI re-derives the numbers quoted in docs/examples/tutorial.md
python scripts/generate_readme_figure.py          # regenerates README output + docs/_static/audit_breast_cancer.png
```

Run scripts from the repo root with `PYTHONPATH=.` unless the package is `pip install -e .`-installed.

## Conventions

- Keep numbers quoted in README/docs reproducible by a script in `scripts/`; CI checks the tutorial ones.
- Anything computed from sampled models is an *inner* approximation (can only understate); say so in
  user-facing text. Coefficient ranges should use `RashomonSet.coef_extremes()` (exact) where affordable.
- `RashomonSet` penalizes the intercept only if `penalize_intercept=True`; `C=np.inf` means λ = 0.
- Line endings are mixed in the index (some files CRLF, some LF; `core.autocrlf=true` locally). When
  rewriting a file programmatically, preserve its existing line endings or the diff becomes the whole file.
- Devnotes under `devnotes/` and `.cognition/` are gitignored local artifacts, not documentation.

## Environment quirks (this Windows machine)

- `python` and `git` are not on PATH in non-interactive shells:
  `%LOCALAPPDATA%\Microsoft\WindowsApps\python.exe` (3.14) and
  `%LOCALAPPDATA%\Programs\Git\cmd\git.exe`.
- Git reports "dubious ownership" for this checkout; use
  `git -c safe.directory=C:/Users/lcawley/projects/StableGLM ...` rather than changing global config.
- GitHub: `fxcawley/StableGLM` (public). The 87 open issues are 3–5× duplicated backlog items created by
  `scripts/gh/sync.ps1`; most are already implemented.
