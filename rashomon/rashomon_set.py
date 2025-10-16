from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
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


class _MembershipOracle:
    """Vectorized penalized objective and membership checks.

    Parameters are shared with the fitted :class:`RashomonSet` instance and the
    oracle exposes helpers that optionally accept pre-computed ``X @ theta``
    values to avoid repeated matrix multiplications (critical for Hit-and-Run
    style sampling in D14).
    """

    __slots__ = (
        "_estimator",
        "_X",
        "_y",
        "_lam",
        "_L_hat",
        "_epsilon",
        "_tol",
        "_n",
        "_d",
    )

    def __init__(
        self,
        *,
        estimator: str,
        X: Array,
        y: Array,
        lam: float,
        L_hat: float,
        epsilon: float,
        tol: float,
    ) -> None:
        self._estimator = estimator
        self._X = X
        self._y = y
        self._lam = float(lam)
        self._L_hat = float(L_hat)
        self._epsilon = float(epsilon)
        self._tol = float(tol)
        self._n = X.shape[0]
        self._d = X.shape[1]

    # ------------------------------------------------------------------ utils
    def with_tolerance(self, tol: float) -> "_MembershipOracle":
        """Return a cloned oracle with a different membership tolerance."""

        return _MembershipOracle(
            estimator=self._estimator,
            X=self._X,
            y=self._y,
            lam=self._lam,
            L_hat=self._L_hat,
            epsilon=self._epsilon,
            tol=tol,
        )

    def _validate_theta(self, theta: Array) -> Array:
        arr = np.asarray(theta, dtype=float)
        if arr.ndim != 1 or arr.shape[0] != self._d:
            raise ValueError("theta must be a 1D vector of length d")
        return arr

    def _validate_thetas(self, Theta: Array) -> Array:
        arr = np.asarray(Theta, dtype=float)
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.ndim != 2 or arr.shape[1] != self._d:
            raise ValueError("Theta must have shape (k, d)")
        return arr

    def _resolve_scores_single(self, theta: Array, x_theta: Optional[Array]) -> Array:
        if x_theta is None:
            return (self._X @ theta).astype(float, copy=False)
        scores = np.asarray(x_theta, dtype=float)
        if scores.ndim == 2:
            scores = scores.reshape(-1)
        if scores.ndim != 1 or scores.shape[0] != self._n:
            raise ValueError("x_theta must have shape (n,)")
        return scores

    def _resolve_scores_many(self, Theta: Array, XTheta: Optional[Array]) -> Array:
        if XTheta is None:
            return (self._X @ Theta.T).astype(float, copy=False)
        scores = np.asarray(XTheta, dtype=float)
        if scores.ndim != 2:
            raise ValueError("XTheta must have shape (n, k) or (k, n)")
        if scores.shape == (Theta.shape[0], self._n):
            scores = scores.T
        elif scores.shape != (self._n, Theta.shape[0]):
            raise ValueError("XTheta must have shape (n, k) or (k, n)")
        return scores

    # -------------------------------------------------------------- objectives
    def objective(self, theta: Array, *, x_theta: Optional[Array] = None) -> float:
        theta_arr = self._validate_theta(theta)
        scores = self._resolve_scores_single(theta_arr, x_theta)
        if self._estimator == "logistic":
            data = float(np.mean(np.logaddexp(0.0, scores) - self._y * scores))
        else:
            resid = self._y - scores
            data = float(0.5 * np.mean(resid * resid))
        reg = 0.5 * self._lam * float(theta_arr @ theta_arr)
        return data + reg

    def objective_many(self, Theta: Array, *, XTheta: Optional[Array] = None) -> Array:
        Theta_arr = self._validate_thetas(Theta)
        scores = self._resolve_scores_many(Theta_arr, XTheta)
        if self._estimator == "logistic":
            loss_terms = np.logaddexp(0.0, scores) - self._y[:, None] * scores
            data = np.mean(loss_terms, axis=0)
        else:
            resid = self._y[:, None] - scores
            data = 0.5 * np.mean(resid * resid, axis=0)
        reg = 0.5 * self._lam * np.sum(Theta_arr * Theta_arr, axis=1)
        return (data + reg).astype(float, copy=False)

    def loss_gap(self, theta: Array, *, x_theta: Optional[Array] = None) -> float:
        return float(self.objective(theta, x_theta=x_theta) - self._L_hat)

    def loss_gap_many(self, Theta: Array, *, XTheta: Optional[Array] = None) -> Array:
        return self.objective_many(Theta, XTheta=XTheta) - self._L_hat

    def contains(self, theta: Array, *, x_theta: Optional[Array] = None, atol: Optional[float] = None) -> bool:
        tol = self._tol if atol is None else float(atol)
        return bool(self.loss_gap(theta, x_theta=x_theta) <= self._epsilon + tol)

    def contains_many(
        self,
        Theta: Array,
        *,
        XTheta: Optional[Array] = None,
        atol: Optional[float] = None,
    ) -> Array:
        tol = self._tol if atol is None else float(atol)
        gaps = self.loss_gap_many(Theta, XTheta=XTheta)
        return gaps <= self._epsilon + tol


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
    bootstrap_fallback : bool
        If True, use parametric bootstrap calibration for LR_alpha when
        Wilks' preconditions are violated (penalized/high-dim).
    bootstrap_reps : int
        Number of bootstrap replicates for LR_alpha fallback (default 200).
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
        bootstrap_fallback: bool = True,
        bootstrap_reps: int = 200,
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
        self.bootstrap_fallback = bool(bootstrap_fallback)
        self.bootstrap_reps = int(bootstrap_reps)

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

        # Cached for HVP / membership
        self._X: Optional[Array] = None
        self._y: Optional[Array] = None
        self._w_diag: Optional[Array] = None  # logistic weights p(1-p)
        self._oracle: Optional[_MembershipOracle] = None

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
        self._oracle = None
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
        if self._X is None or self._y is None or self._lambda is None or self._L_hat is None or self._epsilon_value is None:
            raise RuntimeError("Model calibration incomplete after fit().")
        self._oracle = _MembershipOracle(
            estimator=self.estimator,
            X=self._X,
            y=self._y,
            lam=self._lambda,
            L_hat=self._L_hat,
            epsilon=self._epsilon_value,
            tol=self.tol,
        )
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

    def objective(self, theta: Array, *, x_theta: Optional[Array] = None) -> float:
        """Compute penalized objective L(θ) at parameter θ.

        L(θ) = average loss + (λ/2)||θ||^2
        """
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        if self._oracle is None:
            raise RuntimeError("Membership oracle not initialized.")
        return float(self._oracle.objective(theta, x_theta=x_theta))

    def loss_gap(self, theta: Array, *, x_theta: Optional[Array] = None) -> float:
        """Return L(θ) - L(θ̂)."""
        if self._oracle is None:
            raise RuntimeError("Call fit() first.")
        return float(self._oracle.loss_gap(theta, x_theta=x_theta))

    def contains(self, theta: Array, atol: float = 1e-12, *, x_theta: Optional[Array] = None) -> bool:
        """Return True if θ is inside the ε-Rashomon set: L(θ) - L(θ̂) ≤ ε."""
        if self._epsilon_value is None:
            raise RuntimeError("Call fit() first.")
        if self._oracle is None:
            raise RuntimeError("Membership oracle not initialized.")
        return bool(self._oracle.contains(theta, x_theta=x_theta, atol=float(atol)))

    # ---------------------- Vectorized membership (D14) ---------------------
    def objective_many(self, Theta: Array, *, XTheta: Optional[Array] = None) -> Array:
        """Vectorized penalized objective for a batch of parameters.

        Parameters
        ----------
        Theta : array-like of shape (k, d)
            Rows are parameter vectors θ_j.

        Returns
        -------
        np.ndarray of shape (k,)
            Values L(θ_j) for each row.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        if self._oracle is None:
            raise RuntimeError("Membership oracle not initialized.")
        return self._oracle.objective_many(Theta, XTheta=XTheta)

    def loss_gap_many(self, Theta: Array, *, XTheta: Optional[Array] = None) -> Array:
        """Vectorized loss gap: L(θ_j) - L(θ̂) for rows of Theta."""
        if self._oracle is None:
            raise RuntimeError("Call fit() first.")
        return self._oracle.loss_gap_many(Theta, XTheta=XTheta)

    def contains_many(self, Theta: Array, atol: float = 1e-12, *, XTheta: Optional[Array] = None) -> Array:
        """Vectorized membership oracle: return mask of rows inside ε-set.

        Returns a boolean array of shape (k,).
        """
        if self._epsilon_value is None:
            raise RuntimeError("Call fit() first.")
        if self._oracle is None:
            raise RuntimeError("Membership oracle not initialized.")
        return self._oracle.contains_many(Theta, XTheta=XTheta, atol=float(atol))

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

    def probability_bands(self, X: Array) -> Array:
        """Per-sample probability bands for logistic models via hacking intervals.

        For each row x, computes an interval [p_min, p_max] where
        p_min = σ(z_min), p_max = σ(z_max), and [z_min, z_max] is the
        hacking interval of s := x in parameter space using the
        H^{-1}-norm ellipsoid certificate.

        Returns an array of shape (n, 2) with columns [p_min, p_max].
        """
        if self.estimator != "logistic":
            raise NotImplementedError("probability_bands is only defined for logistic models")
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[1] != self._d:
            raise ValueError("X must be 2D with d features")
        out = np.empty((X.shape[0], 2), dtype=float)
        for i in range(X.shape[0]):
            s = X[i]
            interval = self.hacking_interval(s)
            zmin, zmax = interval["min"], interval["max"]
            pmin, pmax = float(_sigmoid(zmin)), float(_sigmoid(zmax))
            # Ensure ordering in case of numerical edge equality
            if pmin > pmax:
                pmin, pmax = pmax, pmin
            out[i, 0] = pmin
            out[i, 1] = pmax
        return out

    # ---------------------- VIC-style coefficient intervals -----------------
    def coef_intervals(self, indices: Optional[Array] = None) -> Array:
        """Coefficient-wise intervals under the ellipsoid certificate.

        For each selected coordinate j, returns [min_j, max_j] where
        min_j,max_j are the hacking interval for s=e_j.

        Parameters
        ----------
        indices : Optional[array-like]
            Indices of coefficients to compute. If None, compute all.

        Returns
        -------
        np.ndarray of shape (m, 2)
            Interval bounds for selected m coordinates in the order given.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        d = self._d
        if indices is None:
            idx = np.arange(d)
        else:
            idx = np.asarray(indices, dtype=int)
            if idx.ndim != 1:
                raise ValueError("indices must be 1D")
            if np.any((idx < 0) | (idx >= d)):
                raise ValueError("indices out of range")
        out = np.empty((idx.shape[0], 2), dtype=float)
        for k, j in enumerate(idx):
            e = np.zeros(d, dtype=float)
            e[j] = 1.0
            iv = self.hacking_interval(e)
            out[k, 0] = iv["min"]
            out[k, 1] = iv["max"]
        return out

    # -------------------------- Ellipsoid sampler ---------------------------
    def _hessian_matrix(self) -> Array:
        if self._X is None or self._lambda is None:
            raise RuntimeError("Fit before requesting Hessian")
        X = self._X
        n = self._n
        lam = self._lambda
        if self.estimator == "logistic":
            if self._w_diag is None:
                raise RuntimeError("Weights unavailable")
            Xw = X * np.sqrt(self._w_diag)[:, None]
            H = (Xw.T @ Xw) / n + lam * np.eye(self._d)
        else:
            H = (X.T @ X) / n + lam * np.eye(self._d)
        return H.astype(float, copy=False)

    def sample(self, n_samples: int = 100, **kwargs: Any) -> Array:
        """Sample parameters using the configured sampler backend."""

        backend = self.sampler.lower()
        if backend == "ellipsoid":
            if kwargs:
                return self.sample_ellipsoid(n_samples=n_samples, **kwargs)
            return self.sample_ellipsoid(n_samples=n_samples)
        if backend == "hitandrun":
            return self.sample_hitandrun(n_samples=n_samples, **kwargs)
        raise ValueError("Unknown sampler. Use 'ellipsoid' or 'hitandrun'.")

    def sample_ellipsoid(self, n_samples: int = 100, random_state: Optional[int] = None) -> Array:
        """Sample uniformly from the ellipsoid (θ-θ̂)^T H (θ-θ̂) ≤ 2ε.

        Uses Cholesky factorization H=L L^T and maps a uniform point in the
        L2 unit ball via θ = θ̂ + L^{-1} (sqrt(2ε) r u), with u unit vector,
        r ~ U(0,1)^{1/d} for uniform-in-ball radius.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        if self._theta_hat is None or self._epsilon_value is None:
            raise RuntimeError("Model not fully initialized.")
        d = self._d
        if n_samples <= 0:
            raise ValueError("n_samples must be positive")
        H = self._hessian_matrix()
        # Ensure positive definiteness (lam>0 already helps). Add tiny jitter if needed.
        jitter = 0.0
        for _ in range(3):
            try:
                L = np.linalg.cholesky(H + jitter * np.eye(d))
                break
            except np.linalg.LinAlgError:
                jitter = 1e-10 if jitter == 0.0 else jitter * 10.0
        else:
            raise RuntimeError("Hessian not SPD even with jitter")
        seed = self._seed if random_state is None else int(random_state)
        rng = np.random.default_rng(seed)
        samples = np.empty((n_samples, d), dtype=float)
        scale = float(np.sqrt(2.0 * self._epsilon_value))
        for i in range(n_samples):
            g = rng.normal(size=d)
            norm = float(np.linalg.norm(g))
            u = g / (norm + 1e-18)
            r = float(rng.random()) ** (1.0 / d)
            y = scale * r * u
            # Solve L z = y → z = L^{-1} y
            z = np.linalg.solve(L, y)
            samples[i] = self._theta_hat + z
        return samples

    def sample_hitandrun(
        self,
        n_samples: int = 100,
        *,
        burnin: int = 200,
        thin: int = 1,
        directions: Optional[str] = None,
        precondition: Optional[bool] = None,
        t_init: float = 1.0,
        growth: float = 2.0,
        max_bracket: float = 1e6,
        max_bisect: int = 50,
        tol: float = 1e-10,
        random_state: Optional[int] = None,
    ) -> Array:
        """Hit-and-Run sampling using bracketed line search with safeguards."""

        if not self._fitted or self._theta_hat is None:
            raise RuntimeError("Call fit() first.")
        if self._oracle is None or self._X is None or self._y is None or self._epsilon_value is None:
            raise RuntimeError("Model not fully initialized.")
        if n_samples <= 0:
            raise ValueError("n_samples must be positive")
        if burnin < 0:
            raise ValueError("burnin must be non-negative")
        if thin <= 0:
            raise ValueError("thin must be positive")
        if t_init <= 0.0:
            raise ValueError("t_init must be positive")
        if growth <= 1.0:
            raise ValueError("growth must be > 1.0")
        if max_bracket <= 0.0:
            raise ValueError("max_bracket must be positive")

        if directions is None:
            dir_mode = "whitened" if self.measure == "lr" else "euclidean"
        else:
            dir_mode = directions.lower()
        if dir_mode not in {"whitened", "euclidean"}:
            raise ValueError("directions must be 'whitened' or 'euclidean'")

        if precondition is None:
            use_precondition = dir_mode == "whitened"
        else:
            use_precondition = bool(precondition)

        oracle = self._oracle
        eps = float(self._epsilon_value)
        theta = self._theta_hat.copy()
        z = self._X @ theta
        seed = self._seed if random_state is None else int(random_state)
        rng = np.random.default_rng(seed)

        total_steps = burnin + n_samples * max(1, thin)
        max_steps = max(total_steps * 20, total_steps + 1)
        samples = np.empty((n_samples, self._d), dtype=float)
        saved = 0
        step = 0
        line_tol = float(tol)
        max_growth_steps = 64

        def make_delta(direction: Array, direction_proj: Array):
            def _delta(t: float) -> float:
                theta_t = theta + t * direction
                z_t = z + t * direction_proj
                return float(oracle.loss_gap(theta_t, x_theta=z_t) - eps)

            return _delta

        def bracket_limit(direction: Array, direction_proj: Array) -> Optional[float]:
            delta_fn = make_delta(direction, direction_proj)
            delta_inside = delta_fn(0.0)
            t = float(t_init)
            delta_outside = delta_fn(t)
            growth_steps = 0
            while delta_outside <= line_tol and abs(t) < max_bracket and growth_steps < max_growth_steps:
                t *= growth
                delta_outside = delta_fn(t)
                growth_steps += 1
            if delta_outside <= line_tol:
                return None
            inside_val = 0.0
            outside_val = t
            f_inside = delta_inside
            f_outside = delta_outside
            for _ in range(max_bisect):
                if abs(f_outside - f_inside) > 1e-18:
                    cand = outside_val - f_outside * (outside_val - inside_val) / (f_outside - f_inside)
                else:
                    cand = 0.5 * (outside_val + inside_val)
                if not np.isfinite(cand):
                    cand = 0.5 * (outside_val + inside_val)
                low, high = (inside_val, outside_val) if inside_val < outside_val else (outside_val, inside_val)
                if not (low < cand < high):
                    cand = 0.5 * (outside_val + inside_val)
                delta_cand = delta_fn(cand)
                if delta_cand > 0.0:
                    outside_val = cand
                    f_outside = delta_cand
                else:
                    inside_val = cand
                    f_inside = delta_cand
                if abs(outside_val - inside_val) <= max(line_tol, 1e-12 * max(1.0, abs(inside_val))):
                    break
            return inside_val

        while saved < n_samples:
            if step >= max_steps:
                raise RuntimeError("Hit-and-Run did not produce enough samples within step cap")
            step += 1

            g = rng.normal(size=self._d)
            if use_precondition:
                v = self._cg_solve(g, tol=self.tol, max_iter=self.max_iter)
            else:
                v = g
            norm_v = float(np.linalg.norm(v))
            if norm_v < 1e-12:
                continue
            v = v / norm_v
            xv = self._X @ v

            limit_pos = bracket_limit(v, xv)
            if limit_pos is None or limit_pos <= 0.0:
                continue
            limit_neg_pos = bracket_limit(-v, -xv)
            if limit_neg_pos is None or limit_neg_pos <= 0.0:
                continue
            t_plus = limit_pos
            t_minus = -limit_neg_pos

            if not np.isfinite(t_plus) or not np.isfinite(t_minus) or t_plus <= t_minus:
                continue

            t = float(rng.uniform(t_minus, t_plus))
            theta = theta + t * v
            z = z + t * xv

            if step > burnin and ((step - burnin) % max(1, thin) == 0):
                samples[saved] = theta
                saved += 1

        return samples

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

    # --------------------------- Sklearn-style utils ------------------------
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        return {
            "estimator": self.estimator,
            "reg": self.reg,
            "C": self.C,
            "epsilon": self.epsilon,
            "epsilon_mode": self.epsilon_mode,
            "sampler": self.sampler,
            "measure": self.measure,
            "random_state": self.random_state,
            "max_iter": self.max_iter,
            "tol": self.tol,
            "safety_override": self.safety_override,
            "bootstrap_fallback": getattr(self, "bootstrap_fallback", True),
            "bootstrap_reps": getattr(self, "bootstrap_reps", 200),
        }

    def set_params(self, **params: Any) -> "RashomonSet":
        for key, value in params.items():
            if not hasattr(self, key):
                raise ValueError(f"Unknown parameter: {key}")
            setattr(self, key, value)
        return self

    def score(self, X: Array, y: Array) -> float:
        """Sklearn-style score: accuracy (logistic) or R^2 (linear)."""
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        X = np.asarray(X)
        y = np.asarray(y)
        if self.estimator == "logistic":
            y_pred = self.predict(X)
            # y may be float {0,1}; convert to int
            return float(np.mean((y_pred == y.astype(int)).astype(float)))
        # linear: R^2
        y_pred = self.predict(X)
        ss_res = float(np.sum((y - y_pred) ** 2))
        y_mean = float(np.mean(y))
        ss_tot = float(np.sum((y - y_mean) ** 2))
        if ss_tot <= 0:
            return 1.0 if ss_res <= 0 else 0.0
        return 1.0 - ss_res / ss_tot

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
            if not _HAS_SCIPY:
                warnings.warn("scipy not available; falling back to percent_loss calibration")
                return 0.05 * float(self._L_hat), None
            if not (0.0 < alpha < 1.0):
                warnings.warn("LR_alpha expects alpha in (0,1); clipping")
                alpha = float(np.clip(alpha, 1e-12, 1 - 1e-12))
            precond_ok = self._lambda is not None and self._lambda == 0.0
            if precond_ok or self.safety_override:
                eps = 0.5 * chi2.ppf(1.0 - alpha, df=self._d) / self._n
                return float(eps), alpha
            if self.bootstrap_fallback:
                epsb = self._bootstrap_lr_alpha(alpha)
                return float(epsb), alpha
            warnings.warn("Wilks preconditions violated (penalized/high-dim). Falling back to percent_loss.")
            return 0.05 * float(self._L_hat), None
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

    # --------------------- Bootstrap LR_alpha fallback (B9b) ----------------
    def _bootstrap_lr_alpha(self, alpha: float) -> float:
        """Parametric bootstrap calibration for LR_alpha.

        Approximates the (1-α) quantile of 2n [L_b(θ̂) - L_b(θ̂_b)] over
        bootstrap resamples y_b ~ Bernoulli(σ(X θ̂)) and returns ε = q/(2n).

        Notes
        -----
        - Only implemented for logistic models.
        - Uses the same λ as the fitted model.
        - Deterministic given random_state.
        """
        if self.estimator != "logistic":
            warnings.warn("Bootstrap LR_alpha implemented for logistic only; using percent_loss fallback")
            return 0.05 * float(self._L_hat)  # type: ignore[arg-type]
        if self._X is None or self._theta_hat is None or self._lambda is None or self._n is None:
            raise RuntimeError("Model not fully initialized for bootstrap")
        X = self._X
        n = self._n
        lam = self._lambda
        theta_hat = self._theta_hat
        z = X @ theta_hat
        p = _sigmoid(z)
        rng = np.random.default_rng(self._seed if self._seed is not None else 123456789)
        reps = max(int(self.bootstrap_reps), 10)
        lr_vals = np.empty(reps, dtype=float)
        for b in range(reps):
            y_b = (rng.random(n) < p).astype(float)
            theta_b, L_hat_b, _w = self._fit_logistic_l2(X, y_b, lam)
            L_b_at_hat = self._logistic_loss(X, y_b, theta_hat, lam)
            lr_b = 2.0 * n * max(L_b_at_hat - L_hat_b, 0.0)
            lr_vals[b] = lr_b
        # empirical quantile
        q = float(np.quantile(lr_vals, 1.0 - alpha))
        eps = q / (2.0 * n)
        return float(eps)

    # ------------------------------ Placeholders ----------------------------
    def variable_importance(self, mode: str = "VIC") -> Any:
        raise NotImplementedError

    def model_class_reliance(self, X: Array, y: Array, n_permutations: int = 16, random_state: Optional[int] = None) -> Dict[str, Array]:
        """Permutation-based model class reliance (MCR).

        For each feature j, compute the change in score when column j is
        permuted across samples, averaged over n_permutations. Returns a
        dictionary with arrays per family: accuracy drop (logistic) or R^2 drop (linear).

        This is a simple baseline to identify informative features and does not
        account for correlation structures.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        X = np.asarray(X)
        y = np.asarray(y)
        if X.ndim != 2 or X.shape[1] != self._d:
            raise ValueError("X must be 2D with d features")
        base = self.score(X, y)
        rng = np.random.default_rng(self._seed if random_state is None else int(random_state))
        drops = np.zeros(self._d, dtype=float)
        for j in range(self._d):
            vals = np.zeros(n_permutations, dtype=float)
            for b in range(n_permutations):
                Xp = X.copy()
                rng.shuffle(Xp[:, j])
                vals[b] = base - self.score(Xp, y)
            drops[j] = float(np.mean(vals))
        return {"feature_importance": drops}

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
