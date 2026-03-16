"""rashomon-py: stability auditing for GLMs.

Public API (v0)
---------------
The v0 API contract covers the symbols listed in ``__all__`` below.
Everything else in this package is internal and may change without notice.

Core workflow::

    from rashomon import RashomonSet

    rs = RashomonSet(estimator="logistic", epsilon=0.03,
                     epsilon_mode="percent_loss").fit(X, y)
    rs.ambiguity(X)
    rs.variable_importance_cloud()
    rs.discrepancy(X)
    rs.plot_vic()

Plotting helpers are available via ``rashomon.plotting``.
"""

from importlib.metadata import PackageNotFoundError, version

from .plotting import plot_ambiguity, plot_discrepancy, plot_vic
from .rashomon_set import RashomonSet

try:
    __version__ = version("rashomon-py")
except PackageNotFoundError:
    __version__ = "0.1.0"

__all__ = [
    # Core class
    "RashomonSet",
    # Plotting helpers
    "plot_vic",
    "plot_ambiguity",
    "plot_discrepancy",
    # Metadata
    "__version__",
]
