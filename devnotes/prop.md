

# Proposal: Quantifying Stability of GLM Interpretations via ε-Rashomon Sets (Rashomon-GLM)

## 0) One-page summary (what / why / deliverables)

**What.** A rigorously grounded, scikit-friendly toolkit that (i) defines and samples the ε-Rashomon set for convex ERM models (linear & logistic GLMs), (ii) exposes **set-level** interpretability objects (VIC, Shapley-VIC, Model Class Reliance), and (iii) computes **predictive multiplicity** metrics (ambiguity, discrepancy, Rashomon capacity). Fast **ellipsoid bounds** + **exact membership** sampling (hit-and-run) ensure both usability and correctness.

**Why now.** GLMs are interpretable but not uniquely identified under correlation/weak signal. Classical CIs quantify sampling uncertainty for parameters, not **design-choice stability** or **per-point multiplicity** over near-optimal models. The literature is fragmented; there is no unified, sklearn-native library.

**Deliverables (12 weeks).**

* `rashomon-py` (pip): GLM core with ε calibration, ellipsoid certificates, exact membership sampling, VIC/MCR/multiplicity APIs, plots, docs, tests.
* Reproducible benchmarks on tabular datasets; paper-style report (workshop/MLOSS-ready).

---

## 1) Questions for PI (please mark answers or constraints)

1. **Datasets**: Any restrictions beyond standard tabular (e.g., COMPAS/credit/clinical)? Are there internal datasets we can use?
2. **Fairness reporting**: Which group metrics (e.g., TPR/FPR gaps, demographic parity) should we include in the stability ranges?
3. **ε calibration defaults**: Acceptable default—percent loss slack vs LR-level mapping? Any mandate to support high-dimensional LR correction in v0.1?
4. **Compute budget**: OK to depend on SciPy + PyTorch CPU for HVPs/Lanczos, or strict NumPy-only?
5. **Scope**: GLMs only in v0.1, with an **adapter** that ingests external ensembles (e.g., JPMC dropout runs) for metrics—good?
6. **Licensing**: MIT (preferred) vs BSD-3? Any institutional requirements?
7. **Venues**: Target NeurIPS/ICML interpretability workshops + JMLR MLOSS for the library—aligned?

---

## 2) Objectives and success criteria

**O1.** Define, calibrate, and expose ε-Rashomon sets for convex ERM GLMs with clear ε semantics.
**Measure:** users can request `epsilon_mode ∈ {percent_loss, LR_alpha, LR_alpha_highdim}` and see the implied α or ε.

**O2.** Provide **closed-form certificates** for linear functionals and **per-point prediction bands** under an ellipsoidal approximation; validate against exact sampling.
**Measure:** empirical extrema from membership-based sampling fall within analytic bounds for small/moderate ε.

**O3.** Implement and document **VIC/Shapley-VIC**, **Model Class Reliance**, **ambiguity/discrepancy**, and **Rashomon capacity**, computed over the set.
**Measure:** APIs return consistent objects across model classes/ensembles; plots reproduce paper figures.

**O4.** Ship an sklearn-clean API with diagnostics (condition numbers, set fidelity, ESS/mixing hints).
**Measure:** ≥90% unit test coverage for core math; end-to-end examples run < 3 minutes on laptop-class hardware.

---

## 3) Formal setup

**Data.** $(x_i,y_i)_{i=1}^n$, $x_i\in\mathbb{R}^d$, $y_i\in\{0,1\}$ (logistic) or $\mathbb{R}$ (linear).

**Loss (regularized average risk).**

$$
L(\theta)=\frac{1}{n}\sum_{i=1}^n \ell(y_i, x_i^\top \theta)\;+\;\frac{\lambda}{2}\|\theta\|_2^2,
$$

with $\ell_{\text{logistic}}(y,z)=\log(1+e^{z})-y z$, $\ell_{\text{sq}}(y,z)=\tfrac12(y-z)^2$.

**Empirical minimizer.** $\hat\theta=\arg\min_\theta L(\theta)$; Hessian $H=\nabla^2 L(\hat\theta)$ (psd; pd if $\lambda>0$).

**ε-Rashomon set.**

$$
\mathcal{R}_\varepsilon=\{\theta:\; L(\theta)\le L(\hat\theta)+\varepsilon\}.
$$

**ε calibration modes.**

* **Percent loss slack**: user supplies $\rho\in(0,1)$; we set $\varepsilon=\rho\cdot L(\hat\theta)$.
* **LR α-level (low-dim)**: for NLL (no 1/n sum), LR uses $2(\text{NLL}(\theta)-\text{NLL}(\hat\theta))\overset{d}\approx \chi^2_d$. Our $L$ is averaged, so

  $$
  2n\,[L(\theta)-L(\hat\theta)]\approx \chi^2_{d,\,1-\alpha}
  \quad\Rightarrow\quad
  \varepsilon(\alpha)=\frac{1}{2n}\chi^2_{d,\,1-\alpha}.
  $$
* **LR α-level (high-dim)**: **\[Stub]** implement and document a rescaling for $d/n\not\ll1$ (e.g., Sur–Candès-style correction). Provide a switch; default to percent slack if preconditions fail.

**Local ellipsoid (second-order).**

$$
L(\hat\theta+\Delta)\approx L(\hat\theta)+\tfrac12\,\Delta^\top H\,\Delta
\quad\Rightarrow\quad
\mathcal{E}_\varepsilon=\{\Delta:\Delta^\top H\,\Delta\le 2\varepsilon\}.
$$

**Trust-region validity bound.** If $\|\nabla^3L\|\le M$ (Hessian-Lipschitz), then

$$
L(\hat\theta+\Delta)-L(\hat\theta)\le \tfrac12\,\Delta^\top H\,\Delta+\tfrac{M}{6}\|\Delta\|_2^3.
$$

Choose radius $r(\varepsilon,\alpha)$ so the cubic term $\le \alpha\varepsilon$. **\[Stub]**: compute practical $M$ bounds (logistic has $p_i(1-p_i)\le 1/4$) and expose a conservative $r$.

---

## 4) Certificates and bounds (fast)

**Linear functional hacking intervals.** For any $s\in\mathbb{R}^d$,

$$
\max_{\theta\in\hat\theta+\mathcal{E}_\varepsilon} s^\top\theta
= s^\top\hat\theta + \sqrt{2\varepsilon}\,\|s\|_{H^{-1}},
\quad
\min = s^\top\hat\theta - \sqrt{2\varepsilon}\,\|s\|_{H^{-1}}.
$$

Compute $\|s\|_{H^{-1}}$ via CG solves with $H$.

**Per-point prediction bands (logistic).** With margin $m_i(\theta)=x_i^\top\theta$,

$$
|m_i(\theta)-m_i(\hat\theta)|\le \sqrt{2\varepsilon}\,\|x_i\|_{H^{-1}},
$$

and $p_i(\theta)=\sigma(m_i(\theta))$ implies

$$
|p_i(\theta)-p_i(\hat\theta)|\le \tfrac14\sqrt{2\varepsilon}\,\|x_i\|_{H^{-1}}
$$

(uses $|\sigma'|\le 1/4$). We also compute exact min/max $p_i$ along the ellipsoid’s extreme direction $H^{-1}s$. **\[Stub]**: implement monotone mapping to get tight $p$-intervals without the Lipschitz relaxation.

**Ambiguity certificate (0/1 at threshold $\tau$).** For $\tau=0.5$ (logit 0), instance $i$ is **ambiguous** iff

$$
\text{interval } [m_i^{\min}, m_i^{\max}] \ni 0,
\quad
m_i^{\max/\min}=\;x_i^\top\hat\theta \pm \sqrt{2\varepsilon}\,\|x_i\|_{H^{-1}}.
$$

Ambiguity rate = fraction of ambiguous points. **Discrepancy** computed analogously via max pairwise disagreement; see §6.

---

## 5) Exact membership sampling (correctness backbone)

**Goal.** Uniform sampling over $\mathcal{R}_\varepsilon$ (w\.r.t. Euclidean volume in the original parameterization or, optionally, the $H^{1/2}$-whitened space—see §9).

**Algorithm: Hit-and-Run (with membership oracle).**

1. Start $\theta_0=\hat\theta$.
2. Draw direction $v\sim \mathcal{N}(0,I)$ or block-coordinate $v\in\mathbb{R}^d$; optional precondition $v\leftarrow H^{-1/2}u$ for near-isotropy. **\[Stub]**: efficient $H^{-1/2}$–vec via Lanczos/Chebyshev.
3. Find chord $[t_-,t_+]$ s.t. $L(\theta_k+t v)\le L(\hat\theta)+\varepsilon$ for $t\in[t_-,t_+]$. Use bracket + safeguarded Newton/secant; reuse cached $Xv$ for logistic.
4. Sample $t\sim \text{Uniform}[t_-,t_+]$; set $\theta_{k+1}=\theta_k+t v$.
5. Burn-in; thin to control autocorrelation.

**Diagnostics.** Report chord lengths, effective sample size (autocorr of linear probes $s^\top \theta$), and **set fidelity** (should be 1.0 after burn-in). Precompute $z=X\hat\theta$ and maintain $Xv$ to make line searches $O(n)$.

---

## 6) Metrics over the set (definitions + plan)

**VIC / Shapley-VIC.** For a base importance functional $I_j(\theta)$ (e.g., permutation risk increase for feature $j$), compute the distribution $\{I_j(\theta):\theta\in\mathcal{R}_\varepsilon\}$ via samples; summarize ranges and “clouds.” **\[Stub]**: include GLM-specific fast relaxations (e.g., quadratic approximations of risk deltas).

**Model Class Reliance (MCR).**

$$
\text{MCR}_j^{\min/\max} = \min/\max_{\theta\in\mathcal{R}_\varepsilon} \big[ R(\theta) - R_j^{\text{perm}}(\theta) \big].
$$

Compute via samples; where convex relaxations exist for GLMs, expose optional analytic bounds. **\[Stub]**: implement a trust-region subproblem for a smooth surrogate of permutation risk.

**Ambiguity & Discrepancy (classification).**

* **Ambiguity**: fraction of $i$ with $\exists \theta,\theta'\in\mathcal{R}_\varepsilon$ s.t. $\hat y_\theta(x_i)\neq \hat y_{\theta'}(x_i)$. Certificate via per-point margin interval crossing (§4).
* **Discrepancy**: max disagreement rate between any two models in the set. Upper-bound by “interval straddle counts”; empirical via sampled pairs. **\[Stub]**: derive tighter bound using extremal pair construction on margins.

**Rashomon Capacity (probabilistic).** **\[Stub]**: implement the probabilistic multiplicity metric faithfully to the original definition (δ-covering or entropy-like formulation of predictive distributions across the set). Provide both empirical (from samples) and analytic (ellipsoid-induced) approximations where possible.

---

## 7) Computational details

* **Hessian-vector products (HVPs).** Logistic: $H v = X^\top W (X v) + \lambda v$, $W=\mathrm{diag}(p_i(1-p_i))$. No explicit $H$.
* **CG/Lanczos.** Use CG for $H^{-1} s$ norms; use Lanczos or Chebyshev for $H^{-1/2} u$. **\[Stub]**: expose tolerance controls and iteration caps; document stability.
* **Separation and conditioning.** Make $\ell_2$ regularization on by default; detect near-separation (tiny minimum margins / exploding weights) and abort with guidance.

---

## 8) API sketch (sklearn-style)

```python
from rashomon import RashomonSet

rs = RashomonSet(
    estimator="logistic",          # or "linear"
    reg="l2", C=1.0,
    epsilon=0.02,
    epsilon_mode="percent_loss",   # or "LR_alpha", "LR_alpha_highdim"
    sampler="ellipsoid",           # validate=True runs a thin hit-and-run chain
    random_state=0
).fit(X_train, y_train)

diag = rs.diagnostics()            # cond(H), implied_alpha, set_fidelity, ESS

hi   = rs.hacking_interval(s)      # closed-form for linear s
vic  = rs.variable_importance("VIC")
mcr  = rs.model_class_reliance("perm")
mul  = rs.multiplicity(["ambiguity", "discrepancy", "capacity"])

rs.plot.vic_cloud()
rs.plot.prediction_bands(X_test[:64])
```

Also: `from_ensemble(models)` to compute the same metrics from any external model collection (grids, seeds, dropout runs).

---

## 9) Measure choice and invariance (be explicit)

We provide two volume notions:

* **Parametric-Euclidean** (default): uniform w\.r.t. Lebesgue measure in $\theta$ within $\mathcal{R}_\varepsilon$.
* **LR-geometry (whitened)**: define $\tilde\theta=H^{1/2}(\theta-\hat\theta)$; sample uniformly in $\tilde\theta$ within $\tilde{\mathcal{R}}_\varepsilon$.
  **\[Decision]**: default to LR-geometry for ellipsoid sampling (better isotropy), but **report** which measure is used; keep both for transparency.

---

## 10) Evaluation plan

**Datasets.** UCI tabular (binary), one credit-risk, one healthcare tabular (public), COMPAS (if allowed).

**Protocols.**

1. **Tightness**: compare analytic bounds vs empirical extrema from exact sampling across ε grid.
2. **Stability**: report VIC/MCR ranges, ambiguity/discrepancy rates, predictive-band widths; include **fairness metric ranges** across the set.
3. **Scaling**: $n\in\{10^3,10^4,10^5\}$, $d\in\{10,100,1000\}$; wall-clock for HVPs, CG solves, hit-and-run ESS/min.
4. **Baselines**: Wald/profile-LR CIs; bootstrap VI; L1-path variability; external small-NN ensemble (via adapter) to show metric parity across classes.

**Success criteria.** (i) Bounds match empirical extrema at small/moderate ε; (ii) meaningful spread appears under collinearity; (iii) toolkit runs quickly and is reproducible.

---

## 11) Risks & mitigations

* **Ellipsoid mismatch for large ε.** Always validate with a thin exact chain; surface set-fidelity.
* **Ill-conditioning / separation.** Default $\ell_2$, condition diagnostics, auto-reduce ε or stop with guidance.
* **High-dim LR calibration uncertainty.** Ship as optional; default to percent slack until validated. **\[Stub]**: add tests against known regimes.
* **User confusion about ε.** Print implied α (or ε) and show sensitivity plots by default.

---

## 12) Timeline (12 weeks, solo-feasible)

* **W1–2**: GLM training, HVPs, CG norms; ε calibration (percent + LR low-dim); ellipsoid certificates; unit tests.
* **W3–5**: Hit-and-run with membership oracle; $H^{1/2}$ preconditioning **\[Stub: fast approx]**; diagnostics (ESS, chords).
* **W6–7**: VIC, MCR implementations + plots; GLM relaxations **\[Stub]**.
* **W8–9**: Ambiguity/discrepancy; Rashomon capacity **\[Stub: faithful implementation]**; docs.
* **W10–11**: Benchmarks & ablations (ε, conditioning, sampler choice); reproducibility.
* **W12**: Paper-style report; `pip` release; tutorials.

---

## 13) Deliverables

* `rashomon-py` (MIT/BSD-3), docs, examples, CI tests.
* Benchmarks repo + plots.
* Short paper (workshop) + MLOSS draft focused on the library.

---

## 14) Appendices (mathematical stubs to be filled)

* **\[Stub] High-dimensional LR correction:** derive or import a rescaling for $2n\Delta L$ distribution under $d/n\nrightarrow 0$; validate via simulation.
* **\[Stub] Efficient $H^{-1/2}$-vector products:** implement Lanczos polynomial approximation with error controls; fall back to Chebyshev on spectrum bounds (estimated via Lanczos).
* **\[Stub] Tight $p$-intervals (logistic):** 1-D optimization along $H^{-1}s$ direction to replace Lipschitz bound; prove that extreme $p$ occurs along the extreme margin direction.
* **\[Stub] MCR convex relaxation (GLM):** formulate a trust-region surrogate for permutation risk under quadratic approximation; provide guarantees or counterexamples.
* **\[Stub] Discrepancy bound:** derive an upper bound using per-point margin intervals that is tighter than naive straddle counting; compare to sampled pairs.
* **\[Stub] Rashomon capacity implementation:** choose exact formalism (δ-covering / entropy-like); prove estimator consistency from samples; expose δ-sensitivity.

---

## 15) Assumptions & scope

* GLMs (linear/logistic) with $\ell_2$ regularization; fixed features and labels; no claim about multiplicity from feature engineering or data curation.
* Metrics are computed over $\mathcal{R}_\varepsilon$ within the chosen measure (declared).
* For large ε, ellipsoid bounds are diagnostic only; correctness requires membership sampling.

---

