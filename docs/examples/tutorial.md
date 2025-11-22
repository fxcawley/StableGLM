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
*   **Consistent Signals**: Unlike the previous example, here most coefficients stay strictly positive or negative. For instance, `mean area` always has a strong positive impact (or negative depending on label encoding).
*   **Variation**: However, there is still spread. `mean area`'s coefficient can vary significantly while maintaining 99% optimality.

## 3. Predictive Multiplicity (Ambiguity)

![](../_static/tutorial_ambiguity.png)

**Interpretation**:
*   Even with >93% accuracy, there are patients (samples) where models disagree.
*   The vertical bars show the range of probabilities. A few samples cross the 0.5 decision boundary. For these patients, the diagnosis is model-dependent—a "Rashomon" ambiguity.

## 4. Model Class Reliance (MCR)

We compute the Minimum and Maximum importance (accuracy drop) for each feature across the set.

| Feature | Min Importance | Mean Importance | Max Importance |
| :--- | :--- | :--- | :--- |
| mean radius | 0.040 | 0.058 | 0.076 |
| mean texture | 0.025 | 0.040 | 0.056 |
| mean smoothness | 0.020 | 0.037 | 0.058 |
| mean area | 0.072 | 0.094 | 0.118 |
| mean concavity | 0.031 | 0.049 | 0.065 |

**Key Findings:**
*   **Indispensability**: Every feature has `Min Importance > 0`. This means **no feature can be dropped** without hurting the model performance beyond the 1% tolerance. Every feature provides unique, necessary information.
*   **Dominant Predictor**: `mean area` is the most critical feature. Even in the model that relies on it the *least* (Min), removing it causes a 7.2% drop in accuracy. In the model that relies on it the *most* (Max), the drop is 11.8%.
*   **Stability**: The range (Max - Min) is relatively narrow for most features (e.g., `mean smoothness` varies from 2.0% to 5.8%), indicating stable importance.

This contrasts with datasets where features are highly redundant (Min ~ 0). Here, the diagnostic signals are robust.
