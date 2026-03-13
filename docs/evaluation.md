# Evaluation

## Scaling Experiments

All experiments run on the UCI Adult Census dataset (n=30,162, d=104 after one-hot
encoding) and the Wisconsin Breast Cancer dataset (n=569, d=10).

### Wall-Clock Timing (Adult, d=104)

| Operation | n=1K | n=5K | n=10K | n=30K |
|:----------|-----:|-----:|------:|------:|
| `fit()` | 0.01s | 0.04s | 0.04s | 0.10s |
| `sample_ellipsoid(200)` | 0.004s | 0.006s | 0.008s | 0.03s |
| `sample_hitandrun(50)` | 0.2s | 0.8s | 1.4s | 3.9s |

Fitting and ellipsoid sampling scale linearly with n. Hit-and-Run is the bottleneck
because each step requires an O(nd) matrix-vector product for the line search.
Memory is dominated by the data matrix: 25.9 MB for the full Adult dataset.

### Full Pipeline (Adult, n=30K, d=104, eps=3%)

| Step | Time | Result |
|:-----|-----:|:-------|
| fit | 0.09s | L_hat = 0.692 |
| ellipsoid 500 | 0.03s | |
| hitandrun 200 | 21.3s | |
| ambiguity (500 instances) | 3.5s | 100% (strong regularization) |
| VIC (200 samples) | 0.004s | |
| capacity | <0.001s | log_vol = -501.8 |

## Ellipsoid Tightness Analysis

The ellipsoidal approximation $L(\hat\theta + \Delta) \approx L(\hat\theta) + \frac{1}{2}\Delta^T H \Delta$
underpins all certificate-based computations (hacking intervals, probability bands, ambiguity bounds).
We compare the **certificate width** (analytic maximum over the ellipsoid) to the **empirical width**
(range observed across Hit-and-Run samples) to assess how tight the approximation is.

### Breast Cancer (n=569, d=10)

| $\epsilon$ | Certificate Width | H&R Empirical Width | Ratio | Ellipsoid Fidelity |
|:----------:|:-----------------:|:-------------------:|:-----:|:------------------:|
| 0.005 | 0.274 | 0.193 | 1.42x | 99% |
| 0.010 | 0.388 | 0.282 | 1.38x | 99% |
| 0.020 | 0.548 | 0.409 | 1.34x | 99% |
| 0.030 | 0.671 | 0.505 | 1.33x | 99% |
| 0.050 | 0.867 | 0.655 | 1.32x | 99% |
| 0.100 | 1.226 | 0.927 | 1.32x | 99% |

**Interpretation.** In moderate dimensions (d=10), the ellipsoidal certificates are
**1.3-1.4x wider** than the true empirical range. This means the certificates are
conservative (all bounds are valid) but not excessively so. The 99% fidelity confirms
that virtually all ellipsoid samples satisfy the true membership oracle.

The ratio is nearly constant across $\epsilon$, which makes sense: the quadratic
approximation error is $O(\|\Delta\|^3)$, so the relative overestimate is $O(\sqrt{\epsilon})$
-- nearly flat over this range.

### Adult Census (n=30K, d=104)

| $\epsilon$ | Certificate Width | H&R Empirical Width | Ratio | Fidelity |
|:----------:|:-----------------:|:-------------------:|:-----:|:--------:|
| 0.005 | 0.058 | 0.002 | 37.7x | 100% |
| 0.010 | 0.082 | 0.002 | 39.4x | 100% |
| 0.020 | 0.117 | 0.003 | 34.6x | 100% |
| 0.030 | 0.143 | 0.005 | 30.2x | 100% |
| 0.050 | 0.184 | 0.007 | 25.4x | 100% |
| 0.100 | 0.260 | 0.013 | 19.8x | 100% |

**Interpretation.** In high dimensions (d=104), the ratio is 20-38x. This does NOT mean
the certificates are wrong -- 100% fidelity confirms they are valid upper bounds.
The gap arises because:

1. **The certificate reports the worst-case direction.** With 104 features, the maximum
   of $\|x_i\|_{H^{-1}}$ across all features and instances can be much larger than
   the typical direction explored by Hit-and-Run.
2. **200 Hit-and-Run samples in 104 dimensions underexplore the set.** The empirical
   width is an underestimate of the true width because the sampler has not visited
   the extremal regions.

**Practical guidance:** In moderate dimensions (d < 30), use certificates for fast,
tight bounds. In high dimensions (d > 50), use certificates as conservative upper
bounds and Hit-and-Run for tighter empirical estimates (with sufficient samples).
