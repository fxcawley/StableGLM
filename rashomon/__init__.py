"""rashomon-py: does your conclusion survive every equally-good model?

Audit a fitted scikit-learn linear model for model multiplicity: whether its
predictions, coefficient signs and feature rankings would change under a different
model that fits the training data about as well.

Quick start::

    from sklearn.linear_model import LogisticRegression
    from rashomon import audit

    model = LogisticRegression().fit(X, y)
    report = audit(model, X, y)
    print(report.summary())
    report.plot()

Public API (v0.2)
-----------------
``audit`` / ``StabilityReport``
    One-call audit of a fitted ``LogisticRegression``, ``Ridge``, ``LinearRegression``
    (or a ``Pipeline`` ending in one).
``RashomonSet``
    Lower-level API: define, sample and query the ε-Rashomon set directly
    (``RashomonSet.from_sklearn`` converts a fitted model).
``plot_vic`` / ``plot_ambiguity`` / ``plot_discrepancy``
    Plotting helpers for ``RashomonSet`` outputs.

Everything else in this package is internal and may change without notice.
"""

from importlib.metadata import PackageNotFoundError, version

from .audit import StabilityReport, audit
from .plotting import plot_ambiguity, plot_discrepancy, plot_vic
from .rashomon_set import RashomonSet

try:
    __version__ = version("rashomon-py")
except PackageNotFoundError:
    __version__ = "0.3.1"

__all__ = [
    # One-call audit
    "audit",
    "StabilityReport",
    # Expert API
    "RashomonSet",
    # Plotting helpers
    "plot_vic",
    "plot_ambiguity",
    "plot_discrepancy",
    # Metadata
    "__version__",
]
