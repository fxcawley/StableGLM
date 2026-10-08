"""The shipped example notebook must run against the current API."""

import contextlib
import io
import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

NOTEBOOK = Path(__file__).parent.parent / "examples" / "stability_audit.ipynb"


def test_stability_audit_notebook_runs():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert len(cells) >= 6
    namespace: dict = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i, cell in enumerate(cells):
            source = "".join(cell["source"])
            with contextlib.redirect_stdout(io.StringIO()):
                exec(compile(source, f"{NOTEBOOK.name}:cell{i}", "exec"), namespace)
    report = namespace["report"]
    assert report.flip_method == "exact"
    assert set(namespace["comp"]["divergence"]) == set(namespace["X"].columns)
