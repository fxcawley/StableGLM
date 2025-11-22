# Interactive Tutorial

This tutorial demonstrates how to use `RashomonSet` to analyze the stability and multiplicity of a logistic regression model.

## 1. Setup and Data Generation

We'll create a synthetic dataset where some features are correlated, leading to model instability.

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rashomon import RashomonSet

# Reproducibility
np.random.seed(42)

# Generate synthetic data (n=200, d=5)
n_samples = 200
n_features = 5
X = np.random.randn(n_samples, n_features)

# Feature 0 and 1 are correlated
X[:, 1] = X[:, 0] * 0.9 + np.random.randn(n_samples) * 0.1

# True coefficients (Feature 2 is irrelevant)
true_theta = np.array([2.0, -1.5, 0.0, 0.5, -0.5])

# Generate labels
logits = X @ true_theta
probs = 1 / (1 + np.exp(-logits))
y = (np.random.rand(n_samples) < probs).astype(float)

print(f"Data shape: {X.shape}")
print(f"Class balance: {np.mean(y):.2%}")
```
*Output:*
```
Data shape: (200, 5)
Class balance: 53.50%
```

## 2. Fit Rashomon Set

We fit the optimal model and define the Rashomon set with $\epsilon = 0.05$ (allowing 5% higher loss than optimal).

```python
# Initialize RashomonSet
# epsilon_mode="percent_loss" means epsilon=0.05 allows 5% deviation in log-loss
rs = RashomonSet(
    estimator="logistic",
    epsilon=0.05,
    epsilon_mode="percent_loss",
    sampler="hitandrun", # More robust for correlated features
    random_state=42
)

rs.fit(X, y)

# Check diagnostics
diag = rs.diagnostics()
print(f"Optimal Loss (L_hat): {diag['L_hat']:.4f}")
print(f"Epsilon (absolute): {diag['epsilon']:.4f}")
```
*Output:*
```
Optimal Loss (L_hat): 0.4213
Epsilon (absolute): 0.0211
```

## 3. Variable Importance Cloud (VIC)

Instead of a single coefficient vector, we look at the distribution of coefficients across the set.

```python
# Visualize VIC
# This samples models from the set and plots their coefficients
rs.plot_vic()
plt.show()
```

![](../_static/tutorial_vic.png)

**Interpretation**:
- **Feature 0 and 1**: Notice the wide spread and correlation. Because they are correlated in `X`, the model can trade weight between them while maintaining similar loss.
- **Feature 2**: Centered near zero (correctly identified as irrelevant).

## 4. Predictive Multiplicity

How often do these "equally good" models disagree on predictions?

```python
# Compute Ambiguity (fraction of samples with conflicting predictions)
amb = rs.ambiguity(X)
print(f"Ambiguity Rate: {amb['ambiguity_rate']:.2%}")

# Compute Discrepancy (max disagreement between two models)
disc = rs.discrepancy(X, n_samples=100)
print(f"Max Discrepancy: {disc['discrepancy_empirical']:.2%}")
```
*Output:*
```
Ambiguity Rate: 12.50%
Max Discrepancy: 8.00%
```

```python
# Visualizing Ambiguity
# Plot the range of predicted probabilities for the first 20 samples
rs.plot_ambiguity(X[:20], y=y[:20])
plt.show()
```

![](../_static/tutorial_ambiguity.png)

**Interpretation**:
The vertical bars show the range of probabilities assigned by models in the Rashomon set. Samples with bars crossing the 0.5 threshold are "ambiguous" — the model choice determines the label.

## 5. Model Class Reliance (MCR)

MCR gives bounds on feature importance.

```python
# Compute MCR using "residual" permutation (handles correlation better)
mcr = rs.model_class_reliance(X, y, perm_mode="residual", n_permutations=10)

import pandas as pd
mcr_df = pd.DataFrame({
    "Feature": [f"x{i}" for i in range(n_features)],
    "Min Importance": mcr["min_importance"],
    "Max Importance": mcr["max_importance"]
})
print(mcr_df)
```

This shows the range of "indispensability" for each feature.
