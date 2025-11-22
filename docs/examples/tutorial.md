# Case Studies

We explore two datasets to demonstrate how StableGLM reveals different types of model stability.

## 1. Robust Signals: Breast Cancer Diagnosis

**Goal**: Classify tumors as Malignant/Benign using geometric features (Wisconsin Breast Cancer dataset).

**Statistical Context**: In medical diagnosis, we often want to know if a feature (like Tumor Area) is *causally* or *robustly* predictive, or if it's just a proxy for something else. StableGLM tests this by searching for a "good" model that *ignores* the feature. If such a model exists (Min Importance ≈ 0), the feature is not strictly necessary.

```python
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from rashomon import RashomonSet

# 1. Load Data
data = load_breast_cancer()
# Select 5 correlated geometric features
X = data.data[:, [0, 1, 4, 3, 6]] 
y = data.target
feature_names = data.feature_names[[0, 1, 4, 3, 6]]

# 2. Preprocess
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Fit Rashomon Set
# epsilon=0.01 (Tight set: 1% loss tolerance)
rs = RashomonSet(
    estimator="logistic",
    epsilon=0.01, 
    epsilon_mode="percent_loss",
    sampler="hitandrun",
    random_state=42,
    safety_override=True,
    C=0.5
)
rs.fit(X_train, y_train)
```

### Variable Importance Cloud (VIC)
The VIC visualizes the posterior distribution of coefficients over the uniform measure on the Rashomon set.

```python
import matplotlib.pyplot as plt
rs.plot_vic(feature_names=feature_names)
plt.show()
```

![](../_static/tutorial_vic.png)

**Interpretation**:
*   **Consistent Signals**: Features like `mean area` and `mean radius` are consistently important. While they can substitute for each other (collinearity), the model *always* needs geometric size info.

### Model Class Reliance (MCR)

MCR computes the range of feature importance across the entire set of "good" models.

| Feature | Min Importance | Mean Importance | Max Importance |
| :--- | :--- | :--- | :--- |
| mean radius | 0.040 | 0.058 | 0.076 |
| mean area | 0.072 | 0.094 | 0.118 |

**Key Insight**:
*   **Min Importance > 0**: Every feature is indispensable. You cannot remove `mean area` without hurting accuracy by at least 7%, even in the most robust model. This statistically implies that `mean area` adds unique information not fully captured by the other 4 features.

---

## 2. Redundant Signals: Credit Scoring

**Goal**: Predict credit default using the **German Credit** dataset (UCI ID 31).
**Context**: Financial data is often noisy, and many features (e.g., history, savings, job) carry overlapping information.

```python
# (Code for loading German Credit - see scripts/generate_tutorial_german_credit.py)
# ...
# rs.fit(X_train, y_train)
```

### Variable Importance Cloud
![](../_static/german_vic.png)

### Model Class Reliance (MCR)

| Feature | Min Importance | Mean Importance | Max Importance |
| :--- | :--- | :--- | :--- |
| checking_status_<0 | 0.002 | 0.018 | 0.035 |
| credit_amount | -0.009 | 0.002 | 0.016 |
| age | -0.008 | 0.006 | 0.017 |

**Key Insight**:
*   **Min Importance < 0**: For features like `age` and `credit_amount`, the minimum importance is negative. This means there exist "good" models in the Rashomon set for which permuting these features actually *improves* (or doesn't hurt) performance.
*   **Statistical Significance**: A Min Importance near or below zero indicates that the null hypothesis ($H_0$: Feature is irrelevant) cannot be rejected for the class of $\epsilon$-optimal models. The signal provided by `age` is redundant given the other predictors.
