# Case Study: When Equally-Good Models Disagree

Standard model evaluation stops at accuracy. A logistic regression scores 95% on held-out data,
bootstrap confidence intervals are tight, and the model ships. But what if an equally-accurate
model gives the *opposite* prediction for 1 in 10 patients?

This case study demonstrates a concrete failure mode of single-model thinking using the
Wisconsin Breast Cancer dataset, and shows how Rashomon set analysis exposes instability
that bootstrap CIs, cross-validation, and standard interpretability tools completely miss.

## Setup

We fit a regularized logistic regression to classify tumors as malignant or benign using
10 geometric features. The model uses L2 regularization and achieves strong predictive
performance. None of what follows is caused by a "bad" model.

```python
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from rashomon import RashomonSet

data = load_breast_cancer()
feature_names = ['radius', 'texture', 'perimeter', 'area', 'smoothness',
                 'compactness', 'concavity', 'concave_pts', 'symmetry', 'fractal_dim']
X = data.data[:, :10]
y = data.target.astype(float)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)

rs = RashomonSet(
    estimator="logistic",
    epsilon=0.03,              # 3% loss tolerance
    epsilon_mode="percent_loss",
    sampler="hitandrun",       # exact membership sampling
    random_state=42,
    C=0.5,
    safety_override=True,
).fit(X_train, y_train)
```

## Part 1: Bootstrap Says "All Clear"

The standard next step is to assess parameter uncertainty via bootstrap resampling.
StableGLM computes both bootstrap CIs and Rashomon set intervals in one call:

```python
comp = rs.compare_to_bootstrap(
    X_train, y_train,
    n_bootstrap=500,
    n_rashomon=500,
    confidence=0.90,
    feature_names=feature_names,
    random_state=42,
)
```

The results are jarring:

| Feature | Bootstrap 90% CI | Rashomon 90% Interval | Width Ratio |
|:--------|:----------------:|:---------------------:|:-----------:|
| radius | [-0.116, -0.103] | [-1.321, +1.096] | **185x** |
| texture | [-0.083, -0.055] | [-1.530, +1.198] | **96x** |
| perimeter | [-0.117, -0.104] | [-1.383, +1.165] | **209x** |
| area | [-0.112, -0.100] | [-1.545, +1.201] | **220x** |
| smoothness | [-0.066, -0.041] | [-1.524, +1.070] | **104x** |
| compactness | [-0.092, -0.074] | [-1.300, +1.345] | **142x** |
| concavity | [-0.110, -0.093] | [-1.384, +1.289] | **159x** |
| concave_pts | [-0.122, -0.112] | [-1.312, +1.156] | **230x** |
| symmetry | [-0.064, -0.040] | [-1.294, +1.539] | **120x** |
| fractal_dim | [-0.003, +0.024] | [-1.225, +1.125] | **85x** |

![Bootstrap CIs vs Rashomon VIC intervals](../_static/bootstrap_vs_vic.png)

**What this means.** Bootstrap says `concave_pts` has a coefficient of about -0.117, known to
within +/-0.005. The Rashomon set says: there exist models scoring within 3% of the optimum
where that same coefficient is +1.16 or -1.31. The bootstrap-certified "tight" estimate is
an artifact of the optimizer's single path through parameter space.

These are not two ways of measuring the same thing. Bootstrap CIs answer: *"How uncertain
are we about the best-fit parameters given sampling noise?"* The Rashomon set answers a
different question: *"How many qualitatively different models perform nearly as well?"*
The width ratio quantifies the gap between these two questions. A ratio of 230x for
`concave_pts` means the space of good models is 230 times wider than statistical uncertainty
would suggest.

## Part 2: Who Gets a Different Diagnosis?

Abstract instability becomes concrete when we ask which patients are affected.
**Ambiguity** measures the fraction of patients for whom some model in the Rashomon set
gives a different classification than the point estimate:

```python
amb = rs.ambiguity(X_test, threshold_mode="fixed", threshold_value=0.5)
print(f"Ambiguous: {amb['n_ambiguous']}/{len(X_test)} ({amb['ambiguity_rate']:.1%})")
# Ambiguous: 18/171 (10.5%)
```

18 out of 171 test patients receive a diagnosis that depends on which equally-good model
the clinician happens to use. For these patients, the margin interval straddles the
decision threshold -- meaning the model's prediction is not a property of the data,
but of an arbitrary optimization choice.

![Predictive ambiguity across test patients](../_static/tutorial_ambiguity.png)

Red points are patients whose margin intervals cross the decision boundary. Their vertical
bars show the range of predictions across equally-good models. Green points have stable
diagnoses regardless of which model is chosen.

**Discrepancy** goes further and measures worst-case disagreement:

```python
disc = rs.discrepancy(X_test, n_samples=200, n_pairs=200, random_state=42)
print(f"Max pair disagreement: {disc['max_pair_disagreement']:.1%}")
# Max pair disagreement: 87.1%
```

Two models that both achieve near-optimal loss can disagree on **87% of test patients**.
This is not pathological. It reflects the geometric reality of the loss surface:
many diverse parameter vectors land in the same loss-level set.

## Part 3: Which Features Are Driving This?

The Variable Importance Cloud (VIC) shows the distribution of each coefficient across the
Rashomon set, rather than a single point estimate:

```python
rs.plot_vic(n_samples=300, feature_names=feature_names, random_state=42)
```

![Variable Importance Cloud](../_static/tutorial_vic.png)

Unlike a confidence interval (which shrinks with more data), VIC intervals reflect the
**geometry of near-optimal models**. A wide VIC for `perimeter` means there are many
high-performing models that weight `perimeter` anywhere from strongly negative to strongly
positive.

**Model Class Reliance (MCR)** formalizes this by computing the min and max
permutation importance across the Rashomon set:

```python
mcr = rs.model_class_reliance(
    X_train, y_train,
    n_permutations=20, n_samples=100,
    sampler="hitandrun", random_state=42,
)
```

| Feature | MCR- | Mean | MCR+ |
|:--------|-----:|-----:|-----:|
| radius | -0.101 | +0.047 | +0.199 |
| texture | -0.104 | +0.033 | +0.198 |
| perimeter | -0.340 | +0.034 | +0.327 |
| area | -0.282 | +0.031 | +0.266 |
| smoothness | -0.115 | +0.006 | +0.147 |
| compactness | -0.130 | +0.052 | +0.275 |
| concavity | -0.277 | -0.007 | +0.197 |
| concave_pts | -0.223 | +0.026 | +0.269 |
| symmetry | -0.103 | -0.013 | +0.135 |
| fractal_dim | -0.052 | +0.011 | +0.072 |

**Every feature has MCR- < 0.** For every one of these features, there exists a
near-optimal model where removing that feature *improves* accuracy. No feature is
indispensable. This is the hallmark of a problem with high predictive multiplicity:
the features carry redundant information, and the optimizer's particular solution is
one of many valid decompositions.

## Implications

**For practitioners:** A model that "passes" standard validation (good accuracy, tight CIs,
significant p-values) may still exhibit massive predictive instability. The 10.5% ambiguity
rate found here means that roughly 1 in 10 diagnostic recommendations is an artifact of
model selection rather than a property of the patient's data. Standard tools do not
detect this.

**For auditors:** If a regulatory body asks "would a different but equally-valid model give
a different answer?", the answer is yes for 10.5% of patients, and the worst-case
disagreement rate between two valid models is 87%. This is information that bootstrap
CIs cannot provide because they measure the wrong kind of uncertainty.

**For researchers:** The 85-230x width ratio between bootstrap CIs and Rashomon intervals
demonstrates that sampling uncertainty and design-choice multiplicity are orthogonal axes
of model instability. Any interpretability analysis that reports only one is incomplete.
