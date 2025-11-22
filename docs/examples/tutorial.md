# Case Study: Breast Cancer Diagnosis

In this tutorial, we analyze the stability of a diagnostic model for Breast Cancer. We use the classic **Wisconsin Breast Cancer** dataset (sklearn).

The goal is to classify tumors as **Malignant** or **Benign** based on geometric features computed from a digitized image of a fine needle aspirate (FNA).

## 1. Problem Setup

We focus on 5 key geometric features:
*   `mean radius`: Size of the tumor.
*   `mean texture`: Variation in gray-scale values.
*   `mean smoothness`: Local variation in radius lengths.
*   `mean area`: Area of the tumor (highly correlated with radius).
*   `mean concavity`: Severity of concave portions of the contour.

We want to know: **Are all these features necessary, or can valid diagnoses be made using different subsets?**

```python
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from rashomon import RashomonSet

# Load Data
data = load_breast_cancer()
X = data.data[:, [0, 1, 4, 3, 6]] # Select 5 features
y = data.target
feature_names = data.feature_names[[0, 1, 4, 3, 6]]

# Scale features (StandardScaler)
# ... code to scale and split ...
```

## 2. Variable Importance Cloud (VIC)

We fit a Rashomon set with $\epsilon=0.01$ (very tight, allowing only 1% loss deviation from optimal) and `safety_override=True` (since the data is nearly separable).

![](../_static/tutorial_vic.png)

**Interpretation**:
*   **Substitution Effect**: Notice `mean radius` and `mean area`. Both have wide intervals that dip near zero. This implies that they are **substitutes**: a model can rely heavily on Radius and ignore Area, or vice-versa, and still be "good."
*   **Consistent Signals**: `mean concavity` also shows variance, but tends to stay positive (or negative depending on encoding).

## 3. Predictive Multiplicity (Ambiguity)

![](../_static/tutorial_ambiguity.png)

**Interpretation**:
*   Even with 93%+ accuracy, there are patients (samples) where models disagree.
*   The vertical bars show the range of probabilities. A few samples cross the 0.5 decision boundary. For these patients, the diagnosis is model-dependent—a "Rashomon" ambiguity.

## 4. Model Class Reliance (MCR)

We compute the Minimum and Maximum importance (accuracy drop) for each feature across the set.

| Feature | Min Importance | Mean Importance | Max Importance |
| :--- | :--- | :--- | :--- |
| mean radius | 0.000 | 0.0006 | 0.057 |
| mean texture | 0.000 | 0.0004 | 0.044 |
| mean smoothness | 0.000 | 0.0004 | 0.038 |
| mean area | 0.000 | 0.0011 | 0.110 |
| mean concavity | 0.000 | 0.0005 | 0.051 |

**Key Finding: Systemic Redundancy**
*   **Min Importance = 0**: Strikingly, the minimum importance for *every* feature is 0. This means that for any given feature, there exists a valid (good) model that effectively ignores it.
*   **Distributed Signal**: The diagnostic signal is so strong and distributed across these correlated geometric features that **no single feature is a single point of failure**.
*   **Max Importance**: However, some models rely heavily on `mean area` (up to 11% accuracy drop if removed). This confirms that `mean area` is a *powerful* predictor, just not a *unique* one.

This analysis proves that the "optimal" feature importance is just one of many valid possibilities.
