# rashomon-py

**Does your conclusion survive every equally-good model?**

Many parameter vectors fit a training set about as well as the one your solver returned. If some of them reverse your conclusion (a coefficient's sign, a feature ranking, a decision for one row), the conclusion depends on which optimum the solver landed on, not on the data. rashomon-py checks this for a fitted scikit-learn logistic or linear regression.

```python
from sklearn.linear_model import LogisticRegression
from rashomon import audit

model = LogisticRegression().fit(X, y)       # your existing model (Pipelines work too)
report = audit(model, X, y)                  # pandas in -> feature names out

print(report.summary())
report.plot()
report.flipped                               # bool mask: rows whose prediction can flip
report.coefficients                          # DataFrame: estimate, low, high, sign_stable
```

Output for the ten "mean" features of the Wisconsin breast-cancer data with the default `LogisticRegression()`:

```
Stability audit: LogisticRegression (classification), n=569, 10 features
Equally-good models: training log-loss within 0.01704 of the optimum 0.1435
  (one standard error of the 5-fold cross-validated log-loss, whose mean is 0.1458; models closer than this cannot be told apart by CV)
Method: hit-and-run sampling of the exact set (with ellipsoid proposals); 2,000 models, min ESS = 387 -> reliable

Predictions (over the sampled models; lower bounds)
  Flip under some equally-good model:           14.4%  (82 of 569)
  Worst single-model disagreement with yours:    5.1%

Coefficients (exact range across all equally-good models)
  feature                  estimate        low       high  sign
  mean radius               -0.9976     -4.634      2.621  UNSTABLE
  mean texture               -1.399     -2.527    -0.4664  stable
  mean perimeter            -0.9124     -4.613      2.775  UNSTABLE
  mean area                  -1.298     -5.061      2.433  UNSTABLE
  ...
Sign stable: 1 of 10 features.
```

![audit plot](https://raw.githubusercontent.com/fxcawley/StableGLM/main/docs/_static/audit_breast_cancer.png)

The model is 95% accurate. Still, one diagnosis in seven is reversed by some model that cross-validation cannot distinguish from it, and only the coefficient on texture keeps its sign across all of them. Radius, perimeter and area are near-duplicates, so the data fixes their combined effect but not how to split it between them. A claim like "tumour radius lowers the odds" would not survive.

## Install

```bash
pip install rashomon-py          # Python 3.9+; depends on numpy, scipy, scikit-learn, pandas, matplotlib
```

## How this differs from a confidence interval

A confidence interval or bootstrap asks how much the estimate would move under a new sample from the population. rashomon-py asks how many different models fit the data you have about equally well, and whether they agree with yours. The first is sampling uncertainty; the second is model multiplicity (Breiman's "Rashomon effect"). A coefficient can have a narrow confidence interval and still change sign across equally-good models when features are collinear, because the loss is nearly flat along the collinear directions. See [Bootstrap, Rashomon, and Bayesian intervals](https://fxcawley.github.io/StableGLM/guide/why_not_bootstrap.html) for a worked comparison.

## What "equally good" means

The tolerance sets how much worse than optimal a model may be and still count. `audit()` accepts four forms and prints the one in use:

| `tolerance=` | Meaning | When to use |
|---|---|---|
| `"cv"` (default) | one standard error of the cross-validated loss; models closer than this cannot be told apart by cross-validation (the one-standard-error rule from glmnet) | data-driven default |
| `0.01` (any float in (0,1)) | models at most 1% worse than optimal on the training loss | a rule that is easy to state; `0.01` to `0.05` are common |
| `"lr"` / `("lr", 0.05)` | the models not rejected by a likelihood-ratio test at level α | unpenalized fits |
| `("absolute", 0.002)` | a loss gap in training-loss units | reproducing a published setting |

The CV default is permissive, so expect larger sets than with a 1% rule. Results at two or three tolerances say more than any single one; see [Choosing the tolerance](https://fxcawley.github.io/StableGLM/guide/choosing_epsilon.html).

## What the report contains

| `report.` | Meaning | Term in the literature |
|---|---|---|
| `flip_rate`, `flipped` | share (and mask) of rows whose predicted label changes under some equally-good model | ambiguity (Marx, Calmon & Ustun 2020) |
| `max_disagreement` | the largest share of rows on which one equally-good model disagrees with yours | discrepancy (Marx et al. 2020) |
| `coefficients` | each coefficient's range across all equally-good models, and whether its sign is stable | variable importance cloud (Dong & Rudin 2020); hacking intervals (Coker, Rudin & King 2021) |
| `prediction_ranges` | per-row range of predicted probability or fitted value | prediction bands over the Rashomon set |
| `predict_ranges(X_new)` | the same for new rows, through your pipeline | |
| `tolerance` | the loss gap defining the set | ε in the ε-Rashomon set (Fisher, Rudin & Dominici 2019) |
| `rashomon_set`, `samples` | the underlying `RashomonSet` and the sampled parameter vectors | |

Coefficient ranges are exact: each is the solution of a convex program over the true set (automatic when `n · d² ≤ 5·10⁶`; force with `exact_ranges=True`). Prediction-level numbers are computed over models sampled from the exact set, so they are lower bounds that tighten as `n_samples` grows. The report states the effective sample size and whether it is reliable.

## Supported models

`LogisticRegression` / `LogisticRegressionCV` (binary; L2 or unpenalized; `class_weight=None`), `Ridge` / `RidgeCV`, `LinearRegression`, and a `Pipeline` whose last step is one of these. The regularization strength and the unpenalized intercept are converted exactly, and the audit checks that your fitted coefficients sit at the optimum of the reconstructed objective. A mismatch is reported.

Not supported: multiclass (planned), L1 and elastic-net penalties (the level set of a non-smooth objective is not a convex set of the same kind), trees, neural networks. The method needs a convex, twice-differentiable loss, so other L2-penalized GLMs (Poisson, multinomial) are possible extensions.

The method works in small to moderate dimension. Hit-and-run sampling mixes well up to a few dozen features. Above about 60 features the report will show a low effective sample size; reduce the dimension or sample longer. Exact coefficient ranges do not depend on sampling.

## Expert API

`RashomonSet` gives direct access to ε calibration (`percent_loss`, `LR_alpha`, `absolute`), the membership oracle, hit-and-run sampling (with optional ellipsoid proposals), the Hessian-ellipsoid approximation, `functional_range` / `coef_extremes`, model class reliance, Shapley-VIC, and bootstrap and Bayesian comparisons. `RashomonSet.from_sklearn(model, X, y, ...)` builds one from a fitted model. `RashomonSet(C=...)` is not scikit-learn's `C`: it is `1/λ` for the mean-loss objective, and scikit-learn's `C` equals `C/n`. `audit` and `from_sklearn` do the conversion.

```python
from rashomon import RashomonSet
rs = RashomonSet.from_sklearn(model, X, y, epsilon=0.02, random_state=0)
rs.functional_range(x_row)          # exact prediction range for one row, on the logit scale
rs.sample_hitandrun(2000, ellipsoid_mix=0.5)
```

## Documentation

- [Quickstart](https://fxcawley.github.io/StableGLM/guide/quickstart.html)
- [When to use this](https://fxcawley.github.io/StableGLM/guide/when_to_use.html)
- [Choosing the tolerance](https://fxcawley.github.io/StableGLM/guide/choosing_epsilon.html)
- [Interpreting instability](https://fxcawley.github.io/StableGLM/guide/interpreting_instability.html)
- [Bootstrap, Rashomon, and Bayesian intervals](https://fxcawley.github.io/StableGLM/guide/why_not_bootstrap.html)
- [Ellipsoid approximation vs sampling](https://fxcawley.github.io/StableGLM/guide/certificates_vs_sampling.html)
- [Tutorial: when equally-good models disagree](https://fxcawley.github.io/StableGLM/examples/tutorial.html)
- [Evaluation on real datasets](https://fxcawley.github.io/StableGLM/evaluation.html)
- [API reference](https://fxcawley.github.io/StableGLM/api/reference.html)

## References

- Breiman, L. (2001). Statistical modeling: The two cultures. *Statistical Science*, 16(3), 199-231.
- Fisher, A., Rudin, C., & Dominici, F. (2019). All models are wrong, but many are useful. *JMLR*, 20(177), 1-81.
- Marx, C., Calmon, F., & Ustun, B. (2020). Predictive multiplicity in classification. *ICML*.
- Dong, J., & Rudin, C. (2020). Exploring the cloud of variable importance for the set of all good models. *Nature Machine Intelligence*, 2, 810-824.
- Coker, B., Rudin, C., & King, G. (2021). A theory of statistical inference for ensuring the robustness of scientific results. *Management Science*, 67(10), 6174-6197.
- Rudin, C. (2019). Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead. *Nature Machine Intelligence*, 1, 206-215.
- Semenova, L., Rudin, C., & Parr, R. (2022). On the existence of simpler machine learning models. *FAccT*.

## License

MIT
