# rashomon-py

Stability auditing for GLMs: test whether predictions, coefficients, and feature-reliance claims survive movement across the set of near-optimal parameter vectors.

For many practical problems, especially those involving correlated or noisy features, many parameter vectors achieve nearly the same loss. This is the Rashomon effect, named by Breiman (2001) after the Kurosawa film in which several witnesses give contradictory but internally consistent accounts of the same event. The difficulty it creates for interpretability is straightforward: if a feature appears important under one near-optimal model but irrelevant under another, the importance ranking is an artifact of which particular optimum the solver happened to find (Fisher, Rudin, & Dominici, 2019).

rashomon-py makes this problem concrete for L2-regularized logistic and linear regression. Rather than examining a single fitted $\hat\theta$, it characterizes the $\varepsilon$-Rashomon set, the set of all parameter vectors whose loss is within $\varepsilon$ of optimal, and computes interpretability and multiplicity metrics over that set. The question shifts from "what did this model learn?" to "what do all near-optimal models agree on?"

## Install

Requires Python 3.9+.

```bash
pip install -e .          # from a local clone
```

Dependencies: numpy, scipy, scikit-learn, matplotlib. Optional: seaborn, tqdm (`pip install -e ".[full]"`).

## Quickstart

```python
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from rashomon import RashomonSet

X, y = load_breast_cancer(return_X_y=True)
X = StandardScaler().fit_transform(X[:, :10])

rs = RashomonSet(estimator="logistic", epsilon=0.03,
                 epsilon_mode="percent_loss", random_state=0).fit(X, y.astype(float))

amb = rs.ambiguity(X, threshold_mode="fixed", threshold_value=0.5)
print(f"Ambiguity: {amb['ambiguity_rate']:.1%}")

vic = rs.variable_importance_cloud(n_samples=200)
print(f"Coefficient std: {vic['std']}")
```

## What the toolkit computes

The toolkit produces several quantities, each measuring a different aspect of explanation stability.

**Ambiguity** is the fraction of instances whose predicted label changes across the Rashomon set (Marx, Calmon, & Ustun, 2020). An ambiguous instance is one whose prediction is a property of the optimization trajectory, not of the data.

**Discrepancy** is the maximum pairwise disagreement rate between any two models in the set (Marx et al., 2020). If discrepancy is 8%, there exist two near-optimal models that give opposite predictions for 8% of instances.

**Coefficient distributions** show the spread of each parameter across the Rashomon set. This is inspired by the Variable Importance Cloud of Dong & Rudin (2020), adapted to the GLM setting where raw coefficients serve as the natural importance measure. A feature whose coefficient changes sign within $\mathcal{R}_\varepsilon$ has unstable importance.

**Model Class Reliance** (MCR) reports the min and max permutation-based importance of each feature across the set (Fisher, Rudin, & Dominici, 2019). If $\text{MCR}^- < 0$ for a feature, there exists a near-optimal model under which that feature is not merely unimportant but actively harmful.

**Prediction bands** give the range of predictions $[p_i^{\min}, p_i^{\max}]$ for each instance. Points with wide bands have predictions that depend on which $\theta$ was selected.

Two computation modes are available: a Hessian-based ellipsoidal approximation (closed-form, milliseconds) around the optimum, and hit-and-run MCMC sampling using the true Rashomon-set membership oracle. The ellipsoidal pathway is a fast screening approximation whose tightness varies with dimensionality; the sampling pathway targets the true sublevel set but should be interpreted through mixing diagnostics such as effective sample size.

## Benchmark results

Results on real datasets at 3% loss tolerance with CV-selected regularization. Details in [docs/evaluation.md](docs/evaluation.md).

| Dataset | d | Ellip. ambiguity | Empirical ambiguity | Ellip/Emp | Min ESS |
|:--------|--:|:---------------:|:-------------------:|:--------:|--------:|
| Breast Cancer PCA-10 | 10 | 36.0% | 27.0% | 1.3x | 203 |
| Breast Cancer Full | 30 | 18.4% | 10.0% | 1.8x | 60 |
| German Credit | 61 | 85.4% | 11.4% | 7.5x | 23 |
| Adult Census | 104 | 79.0% | 9.2% | 8.6x | 3 |

At $d = 10$, the ellipsoidal approximation says 36% and well-mixed hit-and-run sampling estimates 27%, a 1.3x ratio. At $d = 104$, the ellipsoidal approximation says 79% but hit-and-run finds only 9.2%. Both the approximation (conservative) and the hit-and-run estimate (low ESS = 3) should be interpreted with caution at this dimensionality. The practical implication is that for $d \leq 30$ or so, the ellipsoidal pathway is usually informative; for $d > 60$, sampling diagnostics should be reported prominently; and the intermediate range requires judgment.

## Limitations

The toolkit is not a test of statistical significance, a fairness audit, a model-selection procedure, or a replacement for bootstrap or Bayesian uncertainty analysis. It audits model multiplicity: whether conclusions change across near-optimal parameter vectors.

The toolkit supports only L2-regularized logistic and linear regression. This is a scope constraint, not a roadmap item; the underlying mathematics (convex sublevel sets, Hessian-based ellipsoidal approximation) are specific to this setting.

The Hessian-based ellipsoidal approximation grows conservative as $d$ increases. The observed tightness ratio ranges from 1.3x at $d = 10$ to over 8x at $d = 100$, consistent with the expectation that the quadratic approximation becomes less accurate further from the optimum in higher dimensions.

Hit-and-run sampling targets the true sublevel set but requires adequate chain length. At $d = 104$, the effective sample size after 500 draws is 3, which means the chain has not converged. Long chains or dimensionality reduction are necessary for credible MCMC estimates in high dimensions.

All results depend on $\varepsilon$. The Rashomon set is larger for larger $\varepsilon$, and the relationship between $\varepsilon$ and ambiguity is monotone but not linear; there is typically a phase-transition region in which ambiguity increases rapidly. Running a sensitivity analysis across a range of $\varepsilon$ values is more informative than reporting a single number.

## Scope

rashomon-py does not support tree models, neural networks, L1 or elastic-net penalties, or arbitrary sklearn estimators. It does not compute fairness metrics (though the Rashomon set framework is relevant to fairness; see Rudin, 2019). It does not perform model selection; it audits the stability of a model already fitted.

## Documentation

- [Quickstart](docs/guide/quickstart.md)
- [When to use this toolkit](docs/guide/when_to_use.md)
- [Choosing epsilon](docs/guide/choosing_epsilon.md)
- [Ellipsoidal approximations vs sampling](docs/guide/certificates_vs_sampling.md)
- [Interpreting instability](docs/guide/interpreting_instability.md)
- [Bootstrap, Rashomon, and Bayesian intervals](docs/guide/why_not_bootstrap.md)
- [Tutorial: when equally-good models disagree](docs/examples/tutorial.md)
- [Evaluation on real datasets](docs/evaluation.md)
- [API Reference](docs/api/reference.rst)

## References

- Breiman, L. (2001). Statistical modeling: The two cultures. *Statistical Science*, 16(3), 199--231.
- Fisher, A., Rudin, C., & Dominici, F. (2019). All models are wrong, but many are useful: Learning a variable's importance by studying an entire class of prediction models simultaneously. *JMLR*, 20(177), 1--81.
- Marx, C., Calmon, F., & Ustun, B. (2020). Predictive multiplicity in classification. *ICML*.
- Dong, J. & Rudin, C. (2020). Exploring the cloud of variable importance for the set of all good models. *Nature Machine Intelligence*, 2, 810--824.
- Rudin, C. (2019). Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead. *Nature Machine Intelligence*, 1, 206--215.
- Semenova, L., Rudin, C., & Parr, R. (2022). On the existence of simpler machine learning models. *FAccT*.

## License

MIT
