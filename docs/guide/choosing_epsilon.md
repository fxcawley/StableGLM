# Choosing Epsilon

Epsilon defines what "equally good" means. Every output of rashomon-py depends on it.
There is no correct epsilon -- it is a domain decision, not a statistical one.

## What epsilon controls

Larger epsilon = more models in the Rashomon set = more instability reported.
Smaller epsilon = fewer models = less instability. At epsilon = 0, the Rashomon set
is a single point (the optimum) and nothing is unstable. At very large epsilon,
nearly everything is unstable.

The question is: **at what loss tolerance do your conclusions break?**

## Three calibration modes

### `percent_loss` (default)

```python
rs = RashomonSet(epsilon=0.03, epsilon_mode="percent_loss")
```

Sets epsilon = rho * L(theta_hat). A 3% tolerance means the Rashomon set contains
all models whose loss is within 3% of the best achievable loss.

**Guidance:**
- 0.01 (1%) -- strict. Only models very close to the optimum. If instability
  appears here, it is severe.
- 0.03 (3%) -- moderate. A reasonable default for most applications.
- 0.05-0.10 (5-10%) -- permissive. Includes models with meaningfully higher loss.
  Results may overstate instability.

### `LR_alpha`

```python
rs = RashomonSet(epsilon=0.05, epsilon_mode="LR_alpha")
```

Sets epsilon = chi2(d, 1-alpha) / (2n) via Wilks' theorem. Connects the Rashomon
set to classical likelihood ratio confidence regions. At alpha=0.05, the Rashomon
set approximates the 95% LR confidence region for the parameters.

Use this when you want a statistically grounded epsilon, or when communicating
results to an audience familiar with classical inference.

### `LR_alpha_highdim`

Experimental. Applies a high-dimensional correction (Sur-Candes) to the chi-squared
quantile. Use with caution; the correction is approximate.

## Run sensitivity analysis

Do not report results at a single epsilon. Compute the key metrics across a range
and report the full curve:

```python
from rashomon import RashomonSet
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)
X = StandardScaler().fit_transform(X[:, :10])

for eps in [0.005, 0.01, 0.02, 0.03, 0.05, 0.10]:
    rs = RashomonSet(estimator="logistic", epsilon=eps,
                     epsilon_mode="percent_loss", random_state=0).fit(X, y.astype(float))
    amb = rs.ambiguity(X)
    print(f"epsilon={eps:.3f}  ambiguity={amb['ambiguity_rate']:.1%}")
```

The sensitivity curve tells you where the transition from "stable" to "unstable"
happens. That transition point is often more informative than any single number.

## Common mistakes

**Setting epsilon too large and concluding "everything is unstable."**
At epsilon=0.50, half the loss surface is in the Rashomon set. Of course everything
is unstable. That is not a finding about your model; it is a finding about your
epsilon.

**Setting epsilon too small and concluding "everything is stable."**
At epsilon=0.0001, only models nearly identical to the optimum are included. Of
course nothing is unstable. The interesting question is at what tolerance instability
appears.

**Not running sensitivity analysis.**
A single epsilon gives a single number. Without context, it is hard to interpret.
The sensitivity curve gives the full picture.

**Treating epsilon as a hyperparameter to optimize.**
Epsilon is not something to tune for best performance. It is a definition of
"good enough" that should come from domain knowledge. In medical diagnosis, 1%
loss tolerance might be the right bar. In ad click prediction, 10% might be fine.
The choice should be justified by the application, not by the output.
