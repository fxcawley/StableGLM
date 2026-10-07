# Choosing the tolerance (epsilon)

The parameter $\varepsilon$ (the `tolerance` argument of `audit()`) controls the size of the Rashomon set. It determines what counts as "near-optimal" and therefore governs every output of the toolkit. There is no statistically correct value of $\varepsilon$; it is a definition of how much loss tolerance the analyst is willing to accept, and should be chosen with reference to the application domain, not optimized against the data.

## The default in `audit()`: one cross-validation standard error

```python
report = audit(model, X, y)                 # tolerance="cv"
```

`audit()` refits a clone of your model on `cv` folds, records the held-out loss of each fold, and sets $\varepsilon$ to the standard error of the mean held-out loss. Two models whose training losses differ by less than this could not be told apart by cross-validation on your data. This is the same reasoning as the one-standard-error rule used to pick the regularization strength in glmnet.

Two things to keep in mind. The standard error of a mean loss scales like $\sigma_{\text{loss}}/\sqrt{n}$, which on small datasets can be 10% or more of the loss itself, so sets are larger than under a 1% rule. And the rule is applied to the training objective on a fixed dataset, so it is a calibration heuristic, not a hypothesis test. The report prints the resulting $\varepsilon$ next to the optimal loss.

The other `tolerance` forms map onto the calibration modes below: a float is `percent_loss`, `"lr"` / `("lr", alpha)` is `LR_alpha`, and `("absolute", gap)` is `absolute`.

## Profile-likelihood intervals: `("profile", alpha)`

```python
report = audit(model, X, y, tolerance=("profile", 0.05))
```

Sets $\varepsilon = \chi^2_{1,1-\alpha} / (2n)$. For an unpenalized fit, the range of coefficient $j$ over this set is the $(1-\alpha)$ profile-likelihood confidence interval: maximizing $\beta_j$ subject to $L(\theta) \le L(\hat\theta) + \varepsilon$ minimizes the loss over the other coordinates, which is what profiling does. This is the calibration to use when the coefficient ranges should be read as confidence intervals; `tests/test_validation.py` checks it against R's `confint()` on the UCLA admissions data. Under L2 regularization the correspondence is approximate. Note the difference from `"lr"`, which uses $\chi^2_d$ (a joint region for all coefficients) and therefore gives wider per-coefficient ranges.

## What epsilon controls

The $\varepsilon$-Rashomon set is $\mathcal{R}_\varepsilon = \{\theta : L(\theta) \leq L(\hat\theta) + \varepsilon\}$. Larger $\varepsilon$ admits more parameter vectors and produces wider coefficient distributions, higher ambiguity, and larger discrepancy. Smaller $\varepsilon$ restricts the set toward the optimum and reduces all of these. At $\varepsilon = 0$, the set is a single point and no instability is reported; this is trivially true and uninformative.

The relationship between $\varepsilon$ and ambiguity is monotone but not linear. On the Breast Cancer dataset, ambiguity ranges from 8.8% at $\varepsilon = 0.5\%$ to 60.8% at $\varepsilon = 10\%$, with a phase-transition region (roughly 1--5% for this dataset) in which ambiguity increases rapidly. Below this range, the Rashomon set is small enough that most predictions are stable. Above it, the set admits models that are meaningfully different in their predictions.

## Calibration modes of `RashomonSet`

### Absolute

```python
rs = RashomonSet(epsilon=0.002, epsilon_mode="absolute")
```

Sets $\varepsilon$ to the given loss gap directly, in the units of the training objective. This is what `audit(tolerance="cv")` and `tolerance=("absolute", gap)` use internally.

### Percent loss

```python
rs = RashomonSet(epsilon=0.03, epsilon_mode="percent_loss")
```

Sets $\varepsilon = \rho \cdot L(\hat\theta)$. At $\rho = 0.03$, the Rashomon set contains all models whose loss is within 3% of the best achievable loss. This is the default mode and is interpretable without further context: a 3% loss tolerance is a 3% loss tolerance regardless of the dataset.

As a rough guide: $\rho = 0.01$ is strict (instability here is severe), $\rho = 0.03$ is moderate, and $\rho = 0.05$--$0.10$ is permissive and may overstate instability.

### Likelihood-ratio inversion

```python
rs = RashomonSet(epsilon=0.05, epsilon_mode="LR_alpha")
```

Sets $\varepsilon = \chi^2_{d,1-\alpha} / (2n)$ via Wilks' theorem. This connects the Rashomon set to classical likelihood-ratio confidence regions, which provides a natural calibration for audiences familiar with statistical inference. At $\alpha = 0.05$, the Rashomon set approximates the 95% LR confidence region for the parameters.

### High-dimensional correction

```python
rs = RashomonSet(epsilon=0.05, epsilon_mode="LR_alpha_highdim")
```

Applies a correction for the $d/n \not\ll 1$ regime (Sur & Candès, 2019). This is experimental and should be used with caution.

## Sensitivity analysis

Reporting results at a single $\varepsilon$ is less informative than reporting the sensitivity curve. The following computes the flip rate across a range of tolerances:

```python
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from rashomon import audit

data = load_breast_cancer(as_frame=True)
X, y = data.data.iloc[:, :10], data.target
model = make_pipeline(StandardScaler(), LogisticRegression()).fit(X, y)

for tol in [0.005, 0.01, 0.02, 0.03, 0.05, 0.10]:
    report = audit(model, X, y, tolerance=tol, n_samples=500, random_state=0)
    print(f"tolerance={tol:.3f}  flip rate={report.flip_rate:.1%}  "
          f"sign-stable features={len(report.stable_features)}/{report.n_features}")
```

The transition point, the tolerance at which ambiguity begins to increase rapidly, is often more informative than any individual number. On the Breast Cancer dataset, even at a strict 0.5% tolerance, 8.8% of patients have ambiguous diagnoses, which suggests that the multiplicity is not an artifact of a permissive tolerance but a property of the data and model class.

A diagnostic sweep on German Credit at varying regularization strength illustrates that the multiplicity is persistent: ambiguity remains above 80% across a wide range of regularization strengths at 3% loss tolerance, which indicates that the multiplicity is a property of the data, not an artifact of a particular regularization choice.

## Common failure modes in interpretation

Setting $\varepsilon$ too large and concluding that everything is unstable is uninformative; at $\varepsilon = 0.50$, half the loss surface is in the Rashomon set. Setting $\varepsilon$ too small and concluding that everything is stable is trivially true. The sensitivity analysis avoids both of these by showing where the transition occurs.

Treating $\varepsilon$ as a hyperparameter to optimize against data is a category error. It is a definition of "equally good," not a tuning parameter, and should be justified by domain knowledge. In medical diagnosis, a 1% loss tolerance may be the appropriate bar; in less consequential settings, 5% or 10% may be acceptable.

## References

- Semenova, L., Rudin, C., & Parr, R. (2022). On the existence of simpler machine learning models. *FAccT*.
- Sur, P. & Candès, E. (2019). A modern maximum-likelihood theory for high-dimensional logistic regression. *PNAS*, 116(29), 14516--14525.
