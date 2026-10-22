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

The relationship between $\varepsilon$ and ambiguity is monotone but not linear. On the Breast Cancer dataset with scikit-learn's default logistic regression, the exact flip rate goes from 3.9% at a tolerance of 0.5% of the loss to 17.4% at the `"cv"` default (11.9% of the loss), and the number of sign-stable coefficients from 9 of 10 to 1 of 10 (the full sweep is in the {doc}`tutorial <../examples/tutorial>`).

A useful companion to any tolerance is the posterior mass it corresponds to. For an L2 GLM the $\varepsilon$-Rashomon set is, to second order, the highest-posterior-density region of the Laplace posterior with mass $F_{\chi^2_d}(2n\varepsilon)$ (see {doc}`why_not_bootstrap`). On that dataset a tolerance of 1% of the loss is a 0.1% region, 5% of the loss is a 30% region and the `"cv"` default is a 94.6% region. `RashomonSet.compare_to_bayesian()` reports the level as `hpd_level`.

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

Sets $\varepsilon = \rho \cdot L(\hat\theta)$. At $\rho = 0.03$, the Rashomon set contains all models whose loss is within 3% of the best achievable loss. This is the default mode of `RashomonSet`. It is not comparable across datasets: the set's posterior mass is $F_{\chi^2_d}(2n\rho L(\hat\theta))$, which grows with $n$ at fixed $\rho$. One percent of the loss is a region of 0.1% posterior mass on the breast-cancer model ($n = 569$) and would be a large region at $n = 50{,}000$. Report the mass with the percentage, or use a calibration that scales with $n$.

### Likelihood-ratio inversion

```python
rs = RashomonSet(epsilon=0.05, epsilon_mode="LR_alpha")
```

Sets $\varepsilon = \chi^2_{d,1-\alpha} / (2n)$ via Wilks' theorem. This connects the Rashomon set to classical likelihood-ratio confidence regions, which provides a natural calibration for audiences familiar with statistical inference. At $\alpha = 0.05$, the Rashomon set approximates the 95% LR confidence region for the parameters.

Wilks' theorem assumes $d/n \to 0$. When $d/n$ is not small the likelihood-ratio statistic is not $\chi^2_d$ (Sur & Candès, 2019, derive its limit for unpenalized logistic regression), and the package offers no calibration for that regime; use an absolute or percent-of-loss tolerance, or the `"cv"` default of `audit()`. An earlier `LR_alpha_highdim` mode inflated the $\chi^2$ quantile by $1/(1-d/n)$; that factor is not the Sur-Candès correction and the mode was removed.

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

The sweep gives, for each conclusion, the tolerance at which it fails. On the Breast Cancer dataset the sign of the texture coefficient holds at every tolerance in the sweep, the signs of radius and perimeter fail at 1% of the loss and that of area at 2%, and 3.9% of rows can flip at 0.5% of the loss, a region of less than 0.1% posterior mass. Which of these is relevant depends on the claim being made.

## Common failure modes in interpretation

Setting $\varepsilon$ too large and concluding that everything is unstable is uninformative; at $\varepsilon = 0.50$, half the loss surface is in the Rashomon set. Setting $\varepsilon$ too small and concluding that everything is stable is trivially true. The sensitivity analysis avoids both of these by showing where the transition occurs.

Treating $\varepsilon$ as a hyperparameter to optimize against data is a category error. It is a definition of "equally good," not a tuning parameter, and should be justified by domain knowledge. In medical diagnosis, a 1% loss tolerance may be the appropriate bar; in less consequential settings, 5% or 10% may be acceptable.

## References

- Semenova, L., Rudin, C., & Parr, R. (2022). On the existence of simpler machine learning models. *FAccT*.
- Sur, P. & Candès, E. (2019). A modern maximum-likelihood theory for high-dimensional logistic regression. *PNAS*, 116(29), 14516--14525.
