# Bootstrap, Rashomon, and Bayesian intervals

For an L2-penalized GLM the coefficient range over the Rashomon set is related by identities to the Laplace credible interval, the profile-likelihood interval and the bootstrap interval. This page states them. The numbers come from the [tutorial](../examples/tutorial.md), where they are regenerated in CI.

## The three objects

The fitted objective is $L(\theta) = \text{mean loss} + \tfrac{\lambda}{2}\theta^\top M\theta$, with $M$ masking the unpenalized intercept and $\lambda = 1/(Cn)$ for scikit-learn's `C`. $H = \nabla^2 L(\hat\theta)$ is its Hessian at the optimum.

**Laplace posterior.** $nL(\theta)$ is a negative log-posterior: the log-likelihood plus a Gaussian prior with precision $n\lambda$ on the penalized coordinates. Its mode is $\hat\theta$ and its Laplace approximation is $N(\hat\theta, (nH)^{-1})$. With $\lambda = 0$ this is the asymptotic (Wald) distribution of the MLE. `RashomonSet.compare_to_bayesian()` constructs it.

**Rashomon set.** $\mathcal{R}_\varepsilon = \{\theta : L(\theta) \le L(\hat\theta) + \varepsilon\}$. To second order it is $\{\theta : (\theta - \hat\theta)^\top nH (\theta - \hat\theta) \le 2n\varepsilon\}$, the highest-posterior-density region of the Laplace posterior with mass $F_{\chi^2_d}(2n\varepsilon)$. The exact set differs from this ellipsoid only through the non-quadratic part of the loss inside it.

**Bootstrap.** Refitting the penalized estimator on resampled rows gives the sampling distribution of $\hat\theta$, with variance approximately the sandwich $(nH)^{-1} J (nH)^{-1}$, where $J = \sum_i (y_i - p_i)^2 x_i x_i^\top$ is the empirical score covariance (Efron & Tibshirani, 1993). `RashomonSet.compare_to_bootstrap()` runs it.

## How they relate

*Rashomon range vs credible interval.* The range of coefficient $j$ over the ellipsoid is $\hat\theta_j \pm \sqrt{2n\varepsilon}\,\mathrm{sd}_j$, where $\mathrm{sd}_j$ is the marginal posterior standard deviation; the $(1-\alpha)$ credible interval is $\hat\theta_j \pm z_{1-\alpha/2}\,\mathrm{sd}_j$. The ratio $\sqrt{2n\varepsilon}/z_{1-\alpha/2}$ is the same for every coefficient and depends only on the tolerance. On the breast-cancer model with the `"cv"` tolerance, $\sqrt{2n\varepsilon} = 4.40$ and the set is a 94.6% HPD region; the exact ranges are within 4% of $\sqrt{2n\varepsilon}\,\mathrm{sd}_j$ for all ten coefficients.

*Rashomon range vs profile-likelihood interval.* With `tolerance=("profile", alpha)`, $\varepsilon = \chi^2_1(1-\alpha)/(2n)$, and for an unpenalized fit the range of coefficient $j$ is exactly the $(1-\alpha)$ profile-likelihood confidence interval, because maximizing $\theta_j$ subject to the loss constraint minimizes the loss over the other coordinates. `tests/test_validation.py` reproduces R's `confint()` on the UCLA admissions data to 1e-4.

*Rashomon range vs bootstrap interval.* The bootstrap variance $(nH)^{-1}J(nH)^{-1}$ and the posterior variance $(nH)^{-1}$ agree when $\lambda = 0$ and the model is well specified. With a penalty they differ along the directions the penalty controls: the prior precision $n\lambda$ dominates the data's information there, so the posterior is wide, while the estimator moves little between resamples because the penalty holds it in place. On the breast-cancer model the Laplace sd is four to five times the bootstrap sd for radius, perimeter and area, and 25% above it for texture. The Rashomon half-range is a further factor $\sqrt{2n\varepsilon}$ above the Laplace sd.

A ratio of Rashomon width to bootstrap width is therefore the product of a tolerance factor, $\sqrt{2n\varepsilon}/z$, and a penalty factor, the posterior sd over the sampling sd. Reported on its own it does not say which factor it measures. An earlier version of this page reported such ratios, and compared against a Laplace posterior built with prior precision $\lambda$ instead of $n\lambda$, which is not centred at $\hat\theta$; the "Bootstrap $\ll$ Rashomon $\ll$ Bayesian" ordering it gave came from that error.

## What the Rashomon analysis adds

At a given tolerance the Rashomon set is a credible region of known level under the prior implied by the penalty; it is not an additional source of uncertainty. The analysis provides

- exact extremes over the true set, which differ from the closed forms when the loss is not quadratic inside the set (small samples, near-separation, large tolerances); `compare_to_bayesian()` reports the ratio, so the case at hand can be identified;
- the exact per-row flip test, a statement about an explicit competing model rather than a posterior probability;
- tolerances in loss units, with the corresponding posterior mass reported alongside;
- a reading in terms of competing models: "some model that cross-validation cannot distinguish from yours gives this patient the other diagnosis".

Fisher, Rudin & Dominici (2019) and Semenova, Rudin & Parr (2022) describe the Rashomon set as a property of the model class rather than of sampling noise. For an L2 GLM the two descriptions coincide to second order.

## References

- Efron, B. & Tibshirani, R. (1993). *An Introduction to the Bootstrap*. Chapman & Hall.
- Fisher, A., Rudin, C., & Dominici, F. (2019). All models are wrong, but many are useful. *JMLR*, 20(177), 1--81.
- Semenova, L., Rudin, C., & Parr, R. (2022). On the existence of simpler machine learning models. *FAccT*.
