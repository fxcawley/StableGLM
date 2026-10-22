# Interpreting instability

## Ambiguity

Ambiguity is the fraction of instances whose predicted label changes across the Rashomon set (Marx, Calmon, & Ustun, 2020). An instance is ambiguous if its margin interval $[m_i^{\min}, m_i^{\max}]$ straddles the decision threshold, meaning that there exist near-optimal models on both sides of the classification boundary for that instance. The prediction for such an instance is a property of the optimization trajectory, not of the data.

High ambiguity does not indicate a bad model. On the Breast Cancer dataset with scikit-learn's default logistic regression and the `"cv"` tolerance, 17.4% of rows can flip and the model is 95% accurate (see the {doc}`tutorial <../examples/tutorial>`). Accuracy is an average over rows; ambiguity identifies the rows whose prediction is not determined. The ambiguous rows are not only the borderline ones: in that example every row with a predicted probability between 0.2 and 0.8 can flip, and so can 28 rows with more extreme probabilities, because a row's prediction range depends on its leverage along the poorly determined directions as well as on its distance to the threshold.

The practical response to high ambiguity depends on the application. In clinical settings, ambiguous instances are patients whose diagnosis depends on which near-optimal model the clinician's software happened to fit. Flagging these cases for additional review or reporting prediction intervals rather than point predictions are natural responses. Switching to a different model class is generally not indicated, because if the instability is in the data (correlated features, overlapping class boundaries), a different algorithm will face the same fundamental multiplicity.

Zero ambiguity at a given $\varepsilon$ means that every instance receives the same predicted label under every near-optimal model. This is an informative positive finding, though it is worth testing a larger $\varepsilon$ to identify the tolerance at which instability appears (see {doc}`choosing_epsilon`).

## Coefficient distributions

The coefficient distributions (VIC) show the range of each parameter $\theta_j$ across the Rashomon set. A feature with a wide distribution is one where many different weightings are compatible with near-optimal loss. This is common when features are correlated: the optimizer's particular decomposition of prediction into coefficient weights is one of many valid solutions, and the data does not constrain the decomposition.

A coefficient distribution that crosses zero is notable: it means that some near-optimal models assign the feature a positive effect and others a negative one. The sign of the feature's contribution is not determined by the data at this loss tolerance. Dong & Rudin (2020) introduced the Variable Importance Cloud to visualize this phenomenon for tree models; the adaptation here uses raw coefficients in the GLM setting, where they serve as the natural importance measure.

A narrow coefficient distribution, conversely, indicates that the feature's role is tightly constrained across the Rashomon set. Its importance is a robust property of the data, not an artifact of the fit.

## Model Class Reliance

MCR reports the min and max permutation-based importance of each feature over *sampled* models in the Rashomon set (Fisher, Rudin, & Dominici, 2019). The sampled extremes are inner bounds on the true ones. A negative sampled $\text{MCR}^-$ therefore proves something: a specific near-optimal model exists under which permuting the feature improves the loss. A positive sampled $\text{MCR}^-$ proves nothing about the models the sampler did not visit, so "indispensable across all near-optimal models" cannot be read off it. `mcr_analytic` gives a linearized estimate of the extremes; it is neither an inner nor an outer bound.

## Discrepancy

Discrepancy is the maximum pairwise disagreement rate between any two models in the Rashomon set (Marx et al., 2020). If discrepancy is 8%, there exist two models, both achieving near-optimal loss, that give opposite predictions for 8% of instances simultaneously. The package does not compute it exactly. `discrepancy()` returns the largest disagreement among sampled pairs (an inner bound) and the fraction of rows whose ellipsoid margin interval straddles the threshold (an estimate of an upper bound; it is not rigorous, because the ellipsoid is an approximation). A rigorous upper bound is the exact flip rate from `audit()`: two models can disagree on a row only if at least one of them disagrees with the fitted model there.

## What instability does not indicate

High instability does not indicate that the model is poorly specified or that the features are uninformative. It indicates that the data admits multiple valid explanations. This is a property of the problem, not a deficiency of the algorithm. The appropriate response is not to suppress the finding or switch methods but to be transparent about which conclusions are robust and which depend on the particular model selected. As Semenova, Rudin, & Parr (2022) argue, when many models fit the data equally well, the single "best" model is often an arbitrary point in a large pool of roughly equivalent solutions, and understanding this pool is more informative than treating the optimum as unique.

## References

- Fisher, A., Rudin, C., & Dominici, F. (2019). All models are wrong, but many are useful. *JMLR*, 20(177), 1--81.
- Marx, C., Calmon, F., & Ustun, B. (2020). Predictive multiplicity in classification. *ICML*.
- Dong, J. & Rudin, C. (2020). Exploring the cloud of variable importance for the set of all good models. *Nature Machine Intelligence*, 2, 810--824.
- Semenova, L., Rudin, C., & Parr, R. (2022). On the existence of simpler machine learning models. *FAccT*.
