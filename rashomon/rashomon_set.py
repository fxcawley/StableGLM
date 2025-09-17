from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple
import os
import platform
import warnings

import numpy as np

try:  # optional: sklearn for reliable solvers
    from sklearn.linear_model import LogisticRegression, Ridge
    _HAS_SK = True
except Exception:  # pragma: no cover
    _HAS_SK = False

try:  # optional: scipy for chi2 quantiles
    from scipy.stats import chi2  # type: ignore
    _HAS_SCIPY = True
except Exception:  # pragma: no cover
    _HAS_SCIPY = False


Array = np.ndarray


def _sigmoid(z: Array) -> Array:
    return 1.0 / (1.0 + np.exp(-z))


class RashomonSet:
    """Sklearn-style interface for Rashomon-GLM (v0 functional skeleton).

    Parameters
    ----------
    estimator : {"logistic", "linear"}
        Base GLM family.
    reg : {"l2"}
        Regularization type (currently L2 only).
    C : float
        Inverse regularization strength (sklearn semantics). lambda = 1/C.
    epsilon : float
        If epsilon_mode == "percent_loss": interpreted as rho in (0,1).
        If epsilon_mode == "LR_alpha": interpreted as alpha in (0,1).
    epsilon_mode : {"percent_loss", "LR_alpha", "LR_alpha_highdim"}
        Calibration mode for epsilon.
    sampler : {"ellipsoid", "hitandrun"}
        Sampling backend (placeholder in v0).
    measure : {"param", "lr"}
        Volume measure choice (annotated in diagnostics only).
    random_state : Optional[int]
        Random seed for determinism.
    max_iter : int
        Max iterations for internal solvers when sklearn is unavailable.
    tol : float
        Convergence tolerance for internal solvers and CG.
    safety_override : bool
        If True, do not hard-fail on conditioning/separation guardrails.
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
        max_iter: int = 1000,
        tol: float = 1e-6,
        safety_override: bool = False,
    ) -> None:
        self.estimator = estimator
        self.reg = reg
        self.C = float(C)
        self.epsilon = float(epsilon)
        self.epsilon_mode = epsilon_mode
        self.sampler = sampler
        self.measure = measure
        self.random_state = random_state
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.safety_override = bool(safety_override)

        self._fitted = False
        self._n = 0
        self._d = 0
        self._seed = int(random_state) if random_state is not None else None
        if self._seed is not None:
            np.random.seed(self._seed)

        # Learned state
        self._theta_hat: Optional[Array] = None
        self._L_hat: Optional[float] = None
        self._lambda: Optional[float] = None
        self._epsilon_value: Optional[float] = None
        self._implied_alpha: Optional[float] = None

        # Cached for HVP
        self._X: Optional[Array] = None
        self._y: Optional[Array] = None
        self._w_diag: Optional[Array] = None  # logistic weights p(1-p)

    # ------------------------------ Public API ------------------------------
    def fit(self, X: Array, y: Array) -> "RashomonSet":
        """Fit the base GLM and prepare diagnostics and operators.

        Notes
        -----
        - Uses sklearn (if available) for solvers; otherwise falls back to simple
          Newton/gradient-based updates (sufficient for small tests).
        - L(θ) matches proposal: averaged loss + (λ/2)||θ||^2.
        - Guardrails enforce λ>0 and reasonable conditioning unless overridden.
        """
        X = np.asarray(X)
        y = np.asarray(y)
        if X.ndim != 2:
            raise ValueError("X must be 2D array")
        if y.ndim != 1 or y.shape[0] != X.shape[0]:
            raise ValueError("y must be 1D with same n as X")
        n, d = X.shape
        self._n, self._d = n, d
        lam = 1.0 / self.C
        if lam <= 0.0:
            raise ValueError("C must be finite and > 0 (λ=1/C needed for L2 regularization)")
        if self.reg.lower() != "l2":
            raise ValueError("Only L2 regularization is supported in v0")
        self._lambda = lam

        if self.estimator == "logistic":
            theta, L_hat, w_diag = self._fit_logistic_l2(X, y, lam)
            self._w_diag = w_diag
        elif self.estimator == "linear":
            theta, L_hat = self._fit_linear_ridge(X, y, lam)
            self._w_diag = np.ones(n, dtype=float)
        else:
            raise ValueError("estimator must be 'logistic' or 'linear'")

        self._theta_hat = theta
        self._L_hat = float(L_hat)
        self._X = X
        self._y = y

        # Guardrails: conditioning and separation proxies
        kappa_H = self._estimate_hessian_condition_number()
        w_min = float(np.min(self._w_diag)) if self._w_diag is not None else 1.0
        min_signed_margin = self._compute_min_signed_margin()

        if self.estimator == "logistic" and w_min < 1e-6 and not self.safety_override:
            raise RuntimeError(
                "Near-separation detected (min p(1-p) too small). Ensure L2 regularization or override."
            )
        if kappa_H is not None and kappa_H > 1e8 and not self.safety_override:
            raise RuntimeError(
                f"Ill-conditioned Hessian (cond≈{kappa_H:.2e} > 1e8). Consider stronger L2 or feature scaling."
            )

        # Epsilon calibration
        self._epsilon_value, self._implied_alpha = self._calibrate_epsilon()
        self._fitted = True
        return self

    def diagnostics(self) -> Dict[str, Any]:
        if not self._fitted:
            raise RuntimeError("Call fit() before diagnostics().")
        blas_vendor = _detect_blas_vendor()
        diag: Dict[str, Any] = {
            "n": self._n,
            "d": self._d,
            "L_hat": self._L_hat,
            "theta_norm_2": float(np.linalg.norm(self._theta_hat)) if self._theta_hat is not None else None,
            "epsilon": self._epsilon_value,
            "epsilon_mode": self.epsilon_mode,
            "implied_alpha": self._implied_alpha,
            "measure": self.measure,
            "seed": self._seed,
            "blas_vendor": blas_vendor,
            "kappa_H_est": self._estimate_hessian_condition_number(),
            "min_w_diag": float(np.min(self._w_diag)) if self._w_diag is not None else None,
            "min_signed_margin": self._compute_min_signed_margin(),
            "set_fidelity": None,
            "ess_per_min": None,
        }
        return diag

    def hacking_interval(self, s: Array) -> Dict[str, float]:
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        if self._theta_hat is None or self._epsilon_value is None:
            raise RuntimeError("Model not fully initialized.")
        s = np.asarray(s)
        if s.ndim != 1 or s.shape[0] != self._d:
            raise ValueError("s must be a 1D vector of length d")
        # s^T theta_hat ± sqrt(2 ε) ||s||_{H^{-1}}
        center = float(np.dot(s, self._theta_hat))
        hinv_norm = self._hinv_norm(s)
        delta = float(np.sqrt(2.0 * self._epsilon_value) * hinv_norm)
        return {"min": center - delta, "max": center + delta}

    def objective(self, theta: Array) -> float:
        """Compute penalized objective L(θ) at parameter θ.

        L(θ) = average loss + (λ/2)||θ||^2
        """
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        if self._X is None or self._y is None or self._lambda is None:
            raise RuntimeError("Model not fully initialized.")
        theta = np.asarray(theta, dtype=float)
        if theta.ndim != 1 or theta.shape[0] != self._d:
            raise ValueError("theta must be a 1D vector of length d")
        if self.estimator == "logistic":
            return self._logistic_loss(self._X, self._y, theta, self._lambda)
        resid = self._y - self._X @ theta
        return float(0.5 * np.mean(resid ** 2) + 0.5 * self._lambda * float(theta @ theta))

    def loss_gap(self, theta: Array) -> float:
        """Return L(θ) - L(θ̂)."""
        if self._L_hat is None:
            raise RuntimeError("Call fit() first.")
        return float(self.objective(theta) - self._L_hat)

    def contains(self, theta: Array, atol: float = 1e-12) -> bool:
        """Return True if θ is inside the ε-Rashomon set: L(θ) - L(θ̂) ≤ ε."""
        if self._epsilon_value is None:
            raise RuntimeError("Call fit() first.")
        return bool(self.loss_gap(theta) <= self._epsilon_value + float(atol))

    # --------------------------- Predictive API ----------------------------
    def decision_function(self, X: Array) -> Array:
        if not self._fitted or self._theta_hat is None:
            raise RuntimeError("Call fit() first.")
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[1] != self._d:
            raise ValueError("X must be 2D with d features")
        return (X @ self._theta_hat).astype(float, copy=False)

    def predict_proba(self, X: Array) -> Array:
        if self.estimator != "logistic":
            raise NotImplementedError("predict_proba is only defined for logistic models")
        scores = self.decision_function(X)
        p = _sigmoid(scores)
        return np.vstack([1.0 - p, p]).T

    def predict(self, X: Array) -> Array:
        scores = self.decision_function(X)
        if self.estimator == "logistic":
            return (scores > 0.0).astype(int)
        return scores

    # --------------------------- Sklearn-style attrs -----------------------
    @property
    def coef_(self) -> Array:
        if not self._fitted or self._theta_hat is None:
            raise RuntimeError("Call fit() first.")
        return self._theta_hat

    @property
    def n_features_in_(self) -> int:
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        return int(self._d)

    # ------------------------------ Internals -------------------------------
    def _fit_linear_ridge(self, X: Array, y: Array, lam: float) -> Tuple[Array, float]:
        n, d = X.shape
        if _HAS_SK:
            model = Ridge(alpha=lam, fit_intercept=False)
            model.fit(X, y)
            theta = model.coef_.astype(float, copy=False)
        else:
            A = (X.T @ X) / n + lam * np.eye(d)
            b = (X.T @ y) / n
            theta = np.linalg.solve(A, b)
        resid = y - X @ theta
        L = 0.5 * np.mean(resid ** 2) + 0.5 * lam * float(np.dot(theta, theta))
        return theta, L

    def _fit_logistic_l2(self, X: Array, y: Array, lam: float) -> Tuple[Array, float, Array]:
        n, d = X.shape
        if _HAS_SK:
            # Sklearn uses C = 1/lambda, match our no-intercept setup
            model = LogisticRegression(
                penalty="l2", C=1.0 / lam, fit_intercept=False, solver="lbfgs",
                max_iter=self.max_iter, random_state=self.random_state,
            )
            model.fit(X, y.astype(int))
            theta = model.coef_.ravel().astype(float, copy=False)
        else:
            # Simple gradient descent with backtracking (sufficient for tests)
            theta = np.zeros(d, dtype=float)
            t = 1.0
            for _ in range(self.max_iter):
                z = X @ theta
                p = _sigmoid(z)
                g = (X.T @ (p - y)) / n + lam * theta
                gnorm = np.linalg.norm(g)
                if gnorm < self.tol:
                    break
                # backtracking
                t = 1.0
                L_curr = self._logistic_loss(X, y, theta, lam)
                while True:
                    theta_new = theta - t * g
                    L_new = self._logistic_loss(X, y, theta_new, lam)
                    if L_new <= L_curr - 0.5 * t * gnorm * gnorm or t < 1e-8:
                        theta = theta_new
                        break
                    t *= 0.5
        z = X @ theta
        p = _sigmoid(z)
        w_diag = p * (1.0 - p)
        L = self._logistic_loss(X, y, theta, lam)
        return theta, L, w_diag

    @staticmethod
    def _logistic_loss(X: Array, y: Array, theta: Array, lam: float) -> float:
        n = X.shape[0]
        z = X @ theta
        # average logistic loss + L2
        L = np.mean(np.log1p(np.exp(z)) - y * z) + 0.5 * lam * float(np.dot(theta, theta))
        return float(L)

    def _estimate_hessian_condition_number(self) -> Optional[float]:
        if self._X is None or self._w_diag is None or self._lambda is None:
            return None
        X = self._X
        n = self._n
        lam = self._lambda
        w = self._w_diag
        # Weighted design for Hessian: H = X^T W X / n + lam I
        # SVD on sqrt(W) X (for linear, w=1)
        # Avoid building sqrt(W)X explicitly for huge n; fine here.
        try:
            Xw = X * np.sqrt(w)[:, None]
            svals = np.linalg.svd(Xw, compute_uv=False)
            if svals.size == 0:
                return None
            eig_max = (svals[0] ** 2) / n + lam
            eig_min = (svals[-1] ** 2) / n + lam
            if eig_min <= 0:
                return np.inf
            return float(eig_max / eig_min)
        except Exception:
            return None

    def _compute_min_signed_margin(self) -> Optional[float]:
        if self._X is None or self._theta_hat is None or self._y is None:
            return None
        t = 2.0 * self._y - 1.0  # map {0,1} -> {-1, +1}
        m = t * (self._X @ self._theta_hat)
        return float(np.min(m))

    def _calibrate_epsilon(self) -> Tuple[float, Optional[float]]:
        if self._L_hat is None or self._n is None or self._d is None:
            raise RuntimeError("Fit before calibrating epsilon")
        mode = self.epsilon_mode
        if mode == "percent_loss":
            rho = float(self.epsilon)
            if not (0.0 < rho < 1.0):
                warnings.warn("percent_loss expects epsilon in (0,1); clipping")
                rho = float(np.clip(rho, 1e-12, 1 - 1e-12))
            return rho * float(self._L_hat), None
        if mode == "LR_alpha":
            alpha = float(self.epsilon)
            precond_ok = (self._lambda is not None and self._lambda == 0.0)
            # Penalized LR violates Wilks; warn and fallback unless override
            if not _HAS_SCIPY:
                warnings.warn("scipy not available; falling back to percent_loss calibration")
                return 0.05 * float(self._L_hat), None
            if not precond_ok and not self.safety_override:
                warnings.warn("Wilks preconditions violated (penalized/high-dim). Falling back to percent_loss.")
                return 0.05 * float(self._L_hat), None
            if not (0.0 < alpha < 1.0):
                warnings.warn("LR_alpha expects alpha in (0,1); clipping")
                alpha = float(np.clip(alpha, 1e-12, 1 - 1e-12))
            eps = 0.5 * chi2.ppf(1.0 - alpha, df=self._d) / self._n
            return float(eps), alpha
        if mode == "LR_alpha_highdim":
            warnings.warn("LR_alpha_highdim not yet implemented; using percent_loss fallback.")
            return 0.05 * float(self._L_hat), None
        warnings.warn("Unknown epsilon_mode; defaulting to percent_loss 5%")
        return 0.05 * float(self._L_hat), None

    # --------------------------- H, CG, and norms ---------------------------
    def _Hv(self, v: Array) -> Array:
        if self._X is None or self._w_diag is None or self._lambda is None:
            raise RuntimeError("Operators not initialized")
        X = self._X
        n = self._n
        lam = self._lambda
        if self.estimator == "logistic":
            Xv = X @ v
            WXv = self._w_diag * Xv
            return (X.T @ WXv) / n + lam * v
        # linear: W=1
        return (X.T @ (X @ v)) / n + lam * v

    def _cg_solve(self, b: Array, tol: float, max_iter: int) -> Array:
        x = np.zeros_like(b)
        r = b - self._Hv(x)
        p = r.copy()
        rsold = float(r @ r)
        if np.sqrt(rsold) < tol:
            return x
        for _ in range(max_iter):
            Ap = self._Hv(p)
            alpha = rsold / float(p @ Ap + 1e-18)
            x = x + alpha * p
            r = r - alpha * Ap
            rsnew = float(r @ r)
            if np.sqrt(rsnew) < tol:
                break
            p = r + (rsnew / rsold) * p
            rsold = rsnew
        return x

    def _hinv_norm(self, s: Array) -> float:
        z = self._cg_solve(s, tol=self.tol, max_iter=self.max_iter)
        val = float(np.sqrt(max(s @ z, 0.0)))
        return val

    # ------------------------------ Placeholders ----------------------------
    def variable_importance(self, mode: str = "VIC") -> Any:
        raise NotImplementedError

    def model_class_reliance(self, mode: str = "perm") -> Any:
        raise NotImplementedError

    def multiplicity(self, which: Any = ("ambiguity", "discrepancy")) -> Any:
        raise NotImplementedError


def _detect_blas_vendor() -> Optional[str]:
    try:
        import numpy as _np  # noqa: F401
        from numpy import __config__ as _nc  # type: ignore

        info = getattr(_nc, "blas_opt_info", None)
        if isinstance(info, dict) and info:
            libs = info.get("libraries") or []
            if libs:
                return ",".join(libs)
        return os.environ.get("MKL_SERVICE_FORCE_INTEL", None) or platform.system()
    except Exception:  # pragma: no cover
        return None
