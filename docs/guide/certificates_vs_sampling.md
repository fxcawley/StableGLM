# Ellipsoidal approximations, sampling, and exact ranges

The toolkit has three ways to compute quantities over the Rashomon set. `audit()` picks one and labels each number with the method used. This page explains the trade-offs.

## Exact ranges of linear functionals

For a linear functional $s^\top\theta$ (a single coefficient, $s = e_j$, or the logit of one row, $s = x_i$) the extremes over the true set are the solutions of the convex programs $\max / \min \; s^\top\theta$ subject to $L(\theta) \le L(\hat\theta) + \varepsilon$. `RashomonSet.functional_range(s)` solves them. For linear models the set is the Hessian ellipsoid and the closed form applies. For logistic models the maximizer is $\theta(\mu) = \arg\min_\theta L(\theta) - \mu\, s^\top\theta$ for the unique $\mu > 0$ with $L(\theta(\mu)) = L(\hat\theta) + \varepsilon$; $\mu$ is found by safeguarded root finding with warm-started Newton solves. The cost is a few dozen Hessian builds per functional.

`audit()` uses this for coefficient ranges (and so for `sign_stable`) whenever $n \cdot d^2 \le 5\cdot10^6$, so those results do not depend on how many models were sampled. Sampled extremes understate the range: on the breast-cancer example, 4000 hit-and-run draws recover 70 to 85% of the exact coefficient ranges in 11 dimensions.

## Ellipsoidal approximation

Near the optimum, a second-order Taylor expansion of the loss gives an ellipsoidal approximation to the true Rashomon set:

$$\mathcal{E}_\varepsilon = \bigl\{\hat\theta + \Delta : \Delta^\top H \Delta \leq 2\varepsilon\bigr\}$$

where $H = \nabla^2 L(\hat\theta)$ is the Hessian at the optimum. For any linear functional $s^\top\theta$ (a single coefficient, a linear combination corresponding to a prediction at a particular point), the extrema over $\mathcal{E}_\varepsilon$ have closed forms involving $\lVert s \rVert_{H^{-1}}$. This makes coefficient intervals, prediction bands, and ambiguity bounds available in milliseconds regardless of dimensionality.

The ellipsoidal pathway is best understood as a fast Hessian-based screening approximation. The question is how conservative this approximation is. The tightness ratio (ellipsoidal interval width divided by empirical width from hit-and-run sampling) varies with dimensionality in a consistent pattern:

| Dataset | $d$ | Tightness ratio | Assessment |
|:--------|----:|:---------------:|:-----------|
| Breast Cancer PCA-10 | 10 | 1.3--1.6x | Tight. The approximation tracks sampling closely. |
| Breast Cancer Full | 30 | 2.8--3.7x | Reasonably tight. Useful for screening. |
| German Credit | 61 | 4.4--6.2x | Moderate. Useful as a conservative screen. |
| Adult Census | 104 | 8.2--12.3x | Conservative. Sampling diagnostics dominate interpretation. |

This is expected. The ellipsoidal approximation becomes less accurate as the loss surface deviates from quadratic further from the optimum, and higher-dimensional sets have more room for the true sublevel set to differ from the ellipsoidal shape.

## Hit-and-run sampling

For computations over the true (non-ellipsoidal) Rashomon set, the toolkit uses hit-and-run sampling with a membership oracle. Hit-and-run is a Markov chain method that targets approximately uniform samples from a convex body by repeatedly choosing a random direction, computing the chord of the body along that direction, and sampling uniformly on the chord (Lovász & Vempala, 2006).

The samples are used to compute empirical coefficient distributions (VIC), empirical ambiguity and discrepancy, and other quantities that depend on the actual shape of the Rashomon set rather than its ellipsoidal approximation.

The sampling targets the true sublevel set given adequate mixing, but mixing quality depends on the condition number and dimensionality. The effective sample size (ESS) is the relevant diagnostic:

```python
samples = rs.sample_hitandrun(n_samples=1000, random_state=0, ellipsoid_mix=0.5)
diag = rs.compute_sample_diagnostics(samples)
print(f"Min ESS: {diag['ess_per_param'].min():.0f}")
```

Plain hit-and-run moves along one direction per step, so coordinate autocorrelation is about $1 - 1/d$ and the ESS per step is of order $1/d$: about 40 effective draws per 1000 steps at $d = 11$. The `ellipsoid_mix` option (used by `audit()`) replaces a fraction of the steps with independence proposals drawn uniformly from the Hessian ellipsoid. A proposal is accepted when it lies in the set and the current point lies in the ellipsoid. That is the Metropolis-Hastings ratio for a uniform target, so the chain still samples the exact set. In low to moderate dimension most of the ellipsoid lies inside the set (over 80% of its volume on the breast-cancer example), so most proposals are accepted and the ESS per step improves by about a factor of $d$: around 400 effective draws per 1000 steps at $d = 11$. In high dimension proposals are rarely accepted and the chain behaves like plain hit-and-run.

`audit()` labels the result: min ESS above 100 is "reliable", 30 to 100 "fair", below 30 "unreliable". At $d = 104$ with 500 plain hit-and-run samples, ESS is in the single digits and the sampled quantities should not be trusted. Exact coefficient ranges remain valid there.

## Practical guidance

For $d \leq 30$ or so, the ellipsoidal approximation is sufficient for many screening purposes. It is fast, deterministic, and the tightness ratio is small enough that the results are informative. If the ellipsoidal ambiguity estimate is zero, the model is stable under that approximation at this $\varepsilon$ and sampling is a confirmation step rather than a first pass.

For $d > 60$, the ellipsoidal approximation is conservative enough that its value as a point estimate is limited. Hit-and-run sampling is necessary for sharper empirical estimates, but long chains (1000+ samples) are needed for adequate ESS, and the computational cost grows accordingly.

The intermediate range ($30 < d < 60$) requires judgment. Running the ellipsoidal approximation first is always worthwhile because it is cheap. If the ellipsoidal ambiguity estimate is substantially above zero and the application requires precise numbers, supplementing with sampling is advisable.

## References

- Lovász, L. & Vempala, S. (2006). Hit-and-run from a corner. *SIAM Journal on Computing*, 35(4), 985--1005.
