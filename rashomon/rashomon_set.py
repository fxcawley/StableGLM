from __future__ import annotations

from typing import Any, Dict, Optional
import os
import platform
import numpy as np


class RashomonSet:
    """Sklearn-style interface for Rashomon-GLM.

    Parameters
    ----------
    estimator : {"logistic", "linear"}
        Base GLM family.
    reg : {"l2"}
        Regularization type (currently L2 only).
    C : float
        Inverse regularization strength (sklearn semantics).
    epsilon : float
        Rashomon loss slack.
    epsilon_mode : {"percent_loss", "LR_alpha", "LR_alpha_highdim"}
        Calibration mode for epsilon.
    sampler : {"ellipsoid", "hitandrun"}
        Sampling backend.
    measure : {"param", "lr"}
        Volume measure choice.
    random_state : Optional[int]
        Random seed for determinism.
    """

    def __init__(
        self,
        estimator: str = "logistic",
        reg: str = "l2",
        C: float = 1.0,
        epsilon: float = 0.02,
        epsilon_mode: str = "percent_loss",
        sampler: str = "ellipsoid",
        measure: str = "lr",
        random_state: Optional[int] = None,
    ) -> None:
        self.estimator = estimator
        self.reg = reg
        self.C = float(C)
        self.epsilon = float(epsilon)
        self.epsilon_mode = epsilon_mode
        self.sampler = sampler
        self.measure = measure
        self.random_state = random_state
        self._fitted = False
        self._n = 0
        self._d = 0
        self._seed = int(random_state) if random_state is not None else None
        if self._seed is not None:
            np.random.seed(self._seed)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RashomonSet":
        """Fit the base GLM (placeholder) and prepare diagnostics.

        This skeleton records shapes and parameters. Full GLM training
        and Hessian-based machinery will be implemented per plan.
        """
        X = np.asarray(X)
        y = np.asarray(y)
        if X.ndim != 2:
            raise ValueError("X must be 2D array")
        if y.ndim != 1 or y.shape[0] != X.shape[0]:
            raise ValueError("y must be 1D with same n as X")
        self._n, self._d = X.shape
        # Placeholder theta and loss for early docs/tests
        self._theta_hat = np.zeros(self._d, dtype=float)
        self._L_hat = float(np.mean((y - y.mean()) ** 2) / 2.0)
        self._fitted = True
        return self

    def diagnostics(self) -> Dict[str, Any]:
        """Return core diagnostics needed for reproducibility and UX."""
        if not self._fitted:
            raise RuntimeError("Call fit() before diagnostics().")
        blas_vendor = _detect_blas_vendor()
        diag: Dict[str, Any] = {
            "n": self._n,
            "d": self._d,
            "epsilon": self.epsilon,
            "epsilon_mode": self.epsilon_mode,
            "implied_alpha": None,  # to be populated for LR_alpha
            "measure": self.measure,
            "seed": self._seed,
            "blas_vendor": blas_vendor,
            "set_fidelity": None,   # populated after sampler runs
            "ess_per_min": None,    # populated after sampler runs
        }
        return diag

    # Stubs below to be implemented per epics C/D/E
    def hacking_interval(self, s: np.ndarray) -> Dict[str, float]:
        raise NotImplementedError

    def variable_importance(self, mode: str = "VIC") -> Any:
        raise NotImplementedError

    def model_class_reliance(self, mode: str = "perm") -> Any:
        raise NotImplementedError

    def multiplicity(self, which: Any = ("ambiguity", "discrepancy")) -> Any:
        raise NotImplementedError


def _detect_blas_vendor() -> Optional[str]:
    try:
        import numpy as _np
        from numpy import __config__ as _nc  # type: ignore

        info = getattr(_nc, "blas_opt_info", None)
        if isinstance(info, dict) and info:
            libs = info.get("libraries") or []
            if libs:
                return ",".join(libs)
        # fallback: platform and env hints
        return os.environ.get("MKL_SERVICE_FORCE_INTEL", None) or platform.system()
    except Exception:
        return None
