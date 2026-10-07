"""Regenerate the README example output and figure (docs/_static/audit_breast_cancer.png).

Run from the repository root:  python scripts/generate_readme_figure.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from rashomon import audit

data = load_breast_cancer(as_frame=True)
X, y = data.data.iloc[:, :10], data.target  # the ten "mean ..." features

model = make_pipeline(StandardScaler(), LogisticRegression()).fit(X, y)
report = audit(model, X, y, n_samples=2000, random_state=0)
print(report.summary())

out = Path(__file__).resolve().parents[1] / "docs" / "_static" / "audit_breast_cancer.png"
report.plot().savefig(out, dpi=120, bbox_inches="tight")
print(f"\nsaved {out}")
