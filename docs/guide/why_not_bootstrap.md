# Bootstrap, Rashomon, and Bayesian intervals

Standard practice after fitting a model is to compute bootstrap confidence intervals and treat them as a summary of parameter uncertainty. The question is whether this is sufficient, or whether there are important sources of variation that bootstrap does not capture. The Rashomon set framework suggests that there are.

## What bootstrap measures

Bootstrap resampling quantifies sampling uncertainty: how much would $\hat\theta$ change if the data were redrawn from the same population? The resulting confidence intervals have width $O(1/\sqrt{n})$ and converge to zero as the sample grows. This is a well-understood and useful quantity, but it implicitly conditions on the model structure being correct. The bootstrap treats the logistic regression specification, the choice of features, and the regularization strength as given, and asks only about the noise in the data (Efron & Tibshirani, 1993).

## What the Rashomon set measures

The $\varepsilon$-Rashomon set captures a different source of variation: model multiplicity. It contains all parameter vectors that achieve loss within $\varepsilon$ of the optimum on the observed data. The width of this set depends on $\varepsilon$ and on the geometry of the loss surface (in particular, the eigenstructure of the Hessian at the optimum), not on $n$ directly.

This distinction matters because model multiplicity does not necessarily disappear with more data. If the loss surface has flat directions, meaning directions in parameter space along which the Hessian has small eigenvalues, then many qualitatively different parameter vectors remain near-optimal regardless of sample size. Fisher, Rudin, & Dominici (2019) make this point in the context of variable importance: a feature can appear critical under one near-optimal model and irrelevant under another, and this ambiguity is a property of the model class and the data geometry, not of sampling noise.

The distinction is also important for identifying what Semenova, Rudin, & Parr (2022) call the "hidden subjectivity" in model fitting. Bootstrap CIs assume the model structure is correct and only account for noise. A wide Rashomon interval, by contrast, reveals that there are many other models, perhaps using different effective variable weightings, that are just as accurate. Ignoring this leads to overconfidence in a single model's explanation.

## The empirical gap

On the Breast Cancer dataset ($d = 10$, $n = 569$), VIC intervals are 4--11x wider than 90% bootstrap CIs. This is not a calibration error. It reflects that many different parameter vectors fit the data nearly as well, a fact that bootstrap does not capture because it resamples data, not models. The width ratio depends on $\varepsilon$ and $n$; bootstrap CIs shrink as $O(1/\sqrt{n})$ while Rashomon intervals for fixed $\varepsilon$ do not, so the gap grows with sample size.

## The Bayesian baseline

Under a Laplace approximation with Gaussian prior $N(0, (1/\lambda)I)$, the posterior is approximately Gaussian with covariance $H_{\text{post}}^{-1}$. The resulting credible intervals are wider still than the VIC intervals, because the posterior integrates over all plausible parameter values, not just those within $\varepsilon$ of the optimum. The ordering across all features in the Breast Cancer evaluation is consistent:

$$\text{Bootstrap} \ll \text{Rashomon} \ll \text{Bayesian}$$

At a 3% loss tolerance, the Rashomon set is approximately a subset of the Bayesian posterior. All instability flagged by the Rashomon analysis is already captured by Bayesian uncertainty. The difference is in the interpretation: the Rashomon framing is operational (there exist competing models that are equally good) rather than epistemic (our beliefs about $\theta$ are diffuse). For regulatory or auditing contexts, the operational framing is often more useful, because the question "would a different equally-good model give a different answer?" is more directly actionable than "what is our posterior belief about $\theta$?"

## Implications for high-stakes applications

The gap between bootstrap and Rashomon intervals has practical consequences in settings where individual predictions matter. Rudin (2019) argues that in high-stakes domains like medicine and criminal justice, the existence of a large Rashomon set creates both a risk and an opportunity: a risk because the deployed model may be one of many equally accurate alternatives, some of which make different predictions for the same individual; an opportunity because if the Rashomon set is large, it may contain models that are both accurate and interpretable, or both accurate and fair.

rashomon-py does not directly address the fairness question, but the `compare_to_bootstrap` method makes the gap between sampling uncertainty and model multiplicity explicit:

```python
comp = rs.compare_to_bootstrap(
    X, y,
    n_bootstrap=500, n_rashomon=500,
    confidence=0.90,
    feature_names=feature_names,
    random_state=42,
)

for row in comp['comparison']:
    print(f"{row['feature']}: bootstrap={row['bootstrap_width']:.3f}, "
          f"rashomon={row['rashomon_width']:.3f}, ratio={row['width_ratio']:.1f}x")
```

A width ratio near 1 means that sampling uncertainty and model multiplicity are comparable (unusual in practice). A ratio of 5--10x means that the Rashomon set contains models qualitatively more diverse than sampling variation alone would suggest.

## References

- Breiman, L. (2001). Statistical modeling: The two cultures. *Statistical Science*, 16(3), 199--231.
- Efron, B. & Tibshirani, R. (1993). *An Introduction to the Bootstrap*. Chapman & Hall.
- Fisher, A., Rudin, C., & Dominici, F. (2019). All models are wrong, but many are useful. *JMLR*, 20(177), 1--81.
- Marx, C., Calmon, F., & Ustun, B. (2020). Predictive multiplicity in classification. *ICML*.
- Rudin, C. (2019). Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead. *Nature Machine Intelligence*, 1, 206--215.
- Semenova, L., Rudin, C., & Parr, R. (2022). On the existence of simpler machine learning models. *FAccT*.
