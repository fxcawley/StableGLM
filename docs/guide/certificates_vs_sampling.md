# Ellipsoidal approximations, sampling, and exact ranges

The toolkit has three ways to compute quantities over the Rashomon set. `audit()` picks one and labels each number with the method used. This page explains the trade-offs.

## Exact ranges of linear functionals

For a linear functional $s^\top\theta$ (a single coefficient, $s = e_j$, or the logit of one row, $s = x_i$) the extremes over the true set are the solutions of the convex programs $\max / \min \; s^\top\theta$ subject to $L(\theta) \le L(\hat\theta) + \varepsilon$. `RashomonSet.functional_range(s)` solves them. For linear models the set is the Hessian ellipsoid and the closed form applies. For logistic models the maximizer is $\theta(\mu) = \arg\min_\theta L(\theta) - \mu\, s^\top\theta$ for the unique $\mu > 0$ with $L(\theta(\mu)) = L(\hat\theta) + \varepsilon$; $\mu$ is found by safeguarded root finding with warm-started Newton solves. The cost is a few dozen Hessian builds per functional.

`audit()` uses this for coefficient ranges (and so for `sign_stable`) whenever $n \cdot d^2 \le 5\cdot10^6$, so those results do not depend on how many models were sampled. The per-row flip test is exact in the same way: row $i$ can flip iff $\min\{L(\theta) : x_i^\top\theta = \tau\} \le L(\hat\theta) + \varepsilon$, one equality-constrained Newton solve per row (`RashomonSet.can_flip`); rows already flipped by a sampled model are settled without it. Sampled extremes understate the range: on the breast-cancer example, 4000 hit-and-run draws recover 70 to 85% of the exact coefficient ranges in 11 dimensions.

## Ellipsoidal approximation

Near the optimum, a second-order Taylor expansion of the loss gives an ellipsoidal approximation to the true Rashomon set:

$$\mathcal{E}_\varepsilon = \bigl\{\hat\theta + \Delta : \Delta^\top H \Delta \leq 2\varepsilon\bigr\}$$

where $H = \nabla^2 L(\hat\theta)$ is the Hessian at the optimum. For any linear functional $s^\top\theta$ (a single coefficient, a linear combination corresponding to a prediction at a particular point), the extrema over $\mathcal{E}_\varepsilon$ have closed forms involving $\lVert s \rVert_{H^{-1}}$. This makes coefficient intervals, prediction bands, and ambiguity bounds available in milliseconds regardless of dimensionality.

The ellipsoid is an approximation, not a bound: the true set can extend beyond it in some directions and fall short of it in others. In practice it is close. In the {doc}`evaluation <../evaluation>` on four real datasets (d from 10 to 101, tolerances of 1–3% of the loss and the `"cv"` default), the ellipsoid coefficient intervals are within 1% of the exact ranges in most cases and within 24% in the worst coordinate, and its flip screen is within a few points of the exact flip rate. It overestimates when the tolerance is a large fraction of the loss (36% vs 24% flips at a tolerance of 34% of the loss), where the quadratic model of the loss is no longer accurate.

An earlier version of this page reported the ellipsoid as 4–8x too wide at d = 61–104. That comparison used sampled widths as the reference, and sampled widths understate the true range (see below); against the exact ranges the ellipsoid is close at every dimension tested.

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

`audit()` labels the result: min ESS above 100 is "reliable", 30 to 100 "fair", below 30 "unreliable". At $d = 101$ even 2,000 hybrid draws give an ESS of 4–6, and the sampled quantities should not be read as estimates. Exact coefficient ranges and exact flips are unaffected.

Sampled extremes understate even when the chain mixes well. With 2,000 draws the sampled coefficient ranges cover about 75% of the exact range at d = 10, half at d = 30, 40% at d = 61 and 20% at d = 101; sampled flip rates are a third to a quarter of the exact ones on the larger sets. A sample of the set is not a sample of its boundary.

## Practical guidance

Quantities that are extremes over the set (coefficient ranges, sign stability, whether a row can flip) should be computed exactly. `audit()` does this by default whenever the problem size allows, and the costs are modest: under a second for d ≤ 30, under a minute for n = 5,000 and d = 101.

Sampling is the right tool for quantities that are *not* extremes of a linear functional, such as the disagreement of a single model with yours, and for a picture of the whole set (prediction ranges). Read these through the effective sample size; they are lower bounds.

The ellipsoid is a fast screen that is close to exact at tolerances of a few percent of the loss. It is what `method="ellipsoid"` uses in `audit()` and what `hacking_interval` / `coef_intervals` return.

## References

- Lovász, L. & Vempala, S. (2006). Hit-and-run from a corner. *SIAM Journal on Computing*, 35(4), 985--1005.
