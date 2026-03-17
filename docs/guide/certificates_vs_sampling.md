# Certificates and sampling

The toolkit provides two computation modes for exploring the Rashomon set. They differ in speed and precision, and the tradeoff between them depends on the dimensionality of the problem.

## Ellipsoidal certificates

Near the optimum, a second-order Taylor expansion of the loss gives an ellipsoidal approximation to the true Rashomon set:

$$\mathcal{E}_\varepsilon = \bigl\{\hat\theta + \Delta : \Delta^\top H \Delta \leq 2\varepsilon\bigr\}$$

where $H = \nabla^2 L(\hat\theta)$ is the Hessian at the optimum. For any linear functional $s^\top\theta$ (a single coefficient, a linear combination corresponding to a prediction at a particular point), the extrema over $\mathcal{E}_\varepsilon$ have closed forms involving $\lVert s \rVert_{H^{-1}}$. This makes coefficient intervals, prediction bands, and ambiguity bounds available in milliseconds regardless of dimensionality.

The certificates are valid upper bounds at any $d$, because the ellipsoid is an outer approximation of the true (non-ellipsoidal) sublevel set. The question is how conservative this approximation is. The tightness ratio (certificate interval width divided by empirical width from hit-and-run sampling) varies with dimensionality in a consistent pattern:

| Dataset | $d$ | Tightness ratio | Assessment |
|:--------|----:|:---------------:|:-----------|
| Breast Cancer PCA-10 | 10 | 1.3--1.6x | Tight. Certificates are nearly exact. |
| Breast Cancer Full | 30 | 2.8--3.7x | Reasonably tight. Useful for screening. |
| German Credit | 61 | 4.4--6.2x | Moderate. Valid bounds, not tight. |
| Adult Census | 104 | 8.2--12.3x | Conservative. Sampling needed for precision. |

This is expected. The ellipsoidal approximation becomes less accurate as the loss surface deviates from quadratic further from the optimum, and higher-dimensional sets have more room for the true sublevel set to differ from the ellipsoidal shape.

## Hit-and-run sampling

For computations over the true (non-ellipsoidal) Rashomon set, the toolkit uses hit-and-run sampling with a membership oracle. Hit-and-run is a Markov chain method that generates approximately uniform samples from a convex body by repeatedly choosing a random direction, computing the chord of the body along that direction, and sampling uniformly on the chord (Lovász & Vempala, 2006).

The samples are used to compute empirical coefficient distributions (VIC), empirical ambiguity and discrepancy, and other quantities that depend on the actual shape of the Rashomon set rather than its ellipsoidal approximation.

The sampling is asymptotically exact given adequate mixing, but mixing quality depends on the condition number and dimensionality. The effective sample size (ESS) is the relevant diagnostic:

```python
samples = rs.sample_hitandrun(n_samples=1000, random_state=0)
diag = rs.compute_sample_diagnostics(samples)
print(f"Min ESS: {diag['ess_min']:.0f}")
```

At $d = 10$ with 1000 samples, ESS is typically above 200, which is adequate. At $d = 104$ with 500 samples, ESS drops to 3, which means the chain has not converged and the empirical estimates are unreliable. In the intermediate range, ESS between 50 and 200 is generally sufficient for ambiguity and VIC estimates, though not for tail quantities.

## Practical guidance

For $d \leq 30$ or so, the ellipsoidal certificates are sufficient for most purposes. They are fast, deterministic, and the tightness ratio is small enough that the bounds are informative. If the certificate ambiguity is zero, the model is stable at this $\varepsilon$ and there is no need to sample.

For $d > 60$, the certificates are conservative enough that their value as point estimates is limited, though they remain valid as upper bounds. Hit-and-run sampling is necessary for credible empirical estimates, but long chains (1000+ samples) are needed for adequate ESS, and the computational cost grows accordingly.

The intermediate range ($30 < d < 60$) requires judgment. Running certificates first is always worthwhile because they are free. If the certificate ambiguity is substantially above zero and the application requires precise numbers, supplementing with sampling is advisable.

## References

- Lovász, L. & Vempala, S. (2006). Hit-and-run from a corner. *SIAM Journal on Computing*, 35(4), 985--1005.
