# Certificates vs Sampling

rashomon-py provides two computation modes. They answer the same questions but
trade off speed against precision.

## Certificates (ellipsoidal approximation)

Certificates approximate the Rashomon set as an ellipsoid derived from the
Hessian at the optimum. This gives closed-form bounds on:

- Coefficient intervals (per-feature parameter ranges)
- Probability bands (per-instance prediction ranges)
- Ambiguity (fraction of instances with unstable predictions)

**Speed:** milliseconds, regardless of dimensionality.

**Accuracy:** upper bounds. Always valid, but may overestimate instability.
Tightness depends on dimensionality.

## Hit-and-Run MCMC

Hit-and-Run samples parameter vectors from the true Rashomon set (the actual
sublevel set of the loss, not the ellipsoidal approximation). This gives:

- Empirical coefficient distributions (VIC)
- Empirical ambiguity and discrepancy estimates
- Samples you can inspect directly

**Speed:** seconds to minutes, depending on dimensionality and chain length.

**Accuracy:** asymptotically exact, given adequate mixing. Check effective sample
size (ESS) to verify convergence.

## Decision rule

| Dimensionality | Cert/Exact ratio | Recommendation |
|:---------------|:----------------:|:---------------|
| d < 20 | 1.3-1.6x | Certificates are sufficient for most purposes. |
| 20 < d < 60 | 2-8x | Certificates for fast screening; Hit-and-Run for precision. |
| d > 60 | 8-20x | Hit-and-Run necessary for credible estimates. Expect slow mixing. |

**Always run certificates first.** They cost nothing and give immediate bounds.

- If certificate ambiguity is 0%, you are done. The model is stable at this epsilon.
- If certificate ambiguity is > 0% and you need a precise number, run Hit-and-Run.
- If d > 60 and you need precision, allocate long chains (1000+ samples) and check ESS.

## Checking mixing quality

After Hit-and-Run sampling, check convergence:

```python
rs = RashomonSet(estimator="logistic", epsilon=0.03,
                 epsilon_mode="percent_loss", sampler="hitandrun",
                 random_state=0).fit(X, y)
samples = rs.sample_hitandrun(n_samples=1000, random_state=0)
diag = rs.compute_sample_diagnostics(samples)
print(f"Min ESS: {diag['ess_min']:.0f}")
```

**ESS > 100:** adequate for ambiguity/VIC estimates.
**ESS 10-100:** interpret with caution; consider longer chains.
**ESS < 10:** chain has not converged. Results are unreliable. Increase `n_samples`
or reduce dimensionality.

## Practical example

```python
# Fast screening with certificates
cert_amb = rs.ambiguity(X)  # uses ellipsoidal bounds internally
print(f"Certificate ambiguity: {cert_amb['ambiguity_rate']:.1%}")

# If you need precision, sample
samples = rs.sample_hitandrun(n_samples=500, random_state=42)
vic = rs.variable_importance_cloud(n_samples=500, random_state=42)
```

At d=10 on Breast Cancer, certificate says 36% ambiguity; Hit-and-Run
confirms 27%. The 1.3x ratio means the certificate is a useful fast screen
that slightly overstates the problem.

At d=104 on Adult Census, certificate says 79%; Hit-and-Run finds 9.2%.
The 8.6x ratio means you should not trust the certificate as a point estimate,
only as a conservative upper bound.
