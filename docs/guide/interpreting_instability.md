# Interpreting instability

## Ambiguity

Ambiguity is the fraction of instances whose predicted label changes across the Rashomon set (Marx, Calmon, & Ustun, 2020). An instance is ambiguous if its margin interval $[m_i^{\min}, m_i^{\max}]$ straddles the decision threshold, meaning that there exist near-optimal models on both sides of the classification boundary for that instance. The prediction for such an instance is a property of the optimization trajectory, not of the data.

High ambiguity does not indicate a bad model. On the Breast Cancer dataset at 3% loss tolerance, the model achieves 94% accuracy while 23% of patients have ambiguous diagnoses. The two measures capture different things: accuracy reflects average predictive performance, while ambiguity identifies specific instances for which that performance is not uniquely determined.

The practical response to high ambiguity depends on the application. In clinical settings, ambiguous instances are patients whose diagnosis depends on which near-optimal model the clinician's software happened to fit. Flagging these cases for additional review or reporting prediction intervals rather than point predictions are natural responses. Switching to a different model class is generally not indicated, because if the instability is in the data (correlated features, overlapping class boundaries), a different algorithm will face the same fundamental multiplicity.

Zero ambiguity at a given $\varepsilon$ means that every instance receives the same predicted label under every near-optimal model. This is an informative positive finding, though it is worth testing a larger $\varepsilon$ to identify the tolerance at which instability appears (see {doc}`choosing_epsilon`).

## Coefficient distributions

The coefficient distributions (VIC) show the range of each parameter $\theta_j$ across the Rashomon set. A feature with a wide distribution is one where many different weightings are compatible with near-optimal loss. This is common when features are correlated: the optimizer's particular decomposition of prediction into coefficient weights is one of many valid solutions, and the data does not constrain the decomposition.

A coefficient distribution that crosses zero is notable: it means that some near-optimal models assign the feature a positive effect and others a negative one. The sign of the feature's contribution is not determined by the data at this loss tolerance. Dong & Rudin (2020) introduced the Variable Importance Cloud to visualize this phenomenon for tree models; the adaptation here uses raw coefficients in the GLM setting, where they serve as the natural importance measure.

A narrow coefficient distribution, conversely, indicates that the feature's role is tightly constrained across the Rashomon set. Its importance is a robust property of the data, not an artifact of the fit.

## Model Class Reliance

MCR reports the min and max permutation-based importance of each feature across the Rashomon set (Fisher, Rudin, & Dominici, 2019). If $\text{MCR}^- < 0$, there exists a near-optimal model under which the feature is actively harmful by the permutation metric: removing it improves predictive accuracy. If $\text{MCR}^- > 0$, the feature is indispensable across all near-optimal models.

On the Breast Cancer tutorial case, every feature has $\text{MCR}^- < 0$. No feature is indispensable. This does not mean that the features are useless; the $\text{MCR}^+$ values are positive for most features, indicating that each feature can contribute under some near-optimal model. It means that any single feature importance ranking is an artifact of the particular optimum.

## Discrepancy

Discrepancy is the maximum pairwise disagreement rate between any two models in the Rashomon set (Marx et al., 2020). If discrepancy is 8%, there exist two models, both achieving near-optimal loss, that give opposite predictions for 8% of instances simultaneously. This is a stronger statement than ambiguity, which considers each instance independently. The analytic bound (computed from the ellipsoidal approximation) is typically higher than the empirical discrepancy (computed from sampled pairs), because finite sampling rarely finds the extremal model pair.

## What instability does not indicate

High instability does not indicate that the model is poorly specified or that the features are uninformative. It indicates that the data admits multiple valid explanations. This is a property of the problem, not a deficiency of the algorithm. The appropriate response is not to suppress the finding or switch methods but to be transparent about which conclusions are robust and which depend on the particular model selected. As Semenova, Rudin, & Parr (2022) argue, when many models fit the data equally well, the single "best" model is often an arbitrary point in a large pool of roughly equivalent solutions, and understanding this pool is more informative than treating the optimum as unique.

## References

- Fisher, A., Rudin, C., & Dominici, F. (2019). All models are wrong, but many are useful. *JMLR*, 20(177), 1--81.
- Marx, C., Calmon, F., & Ustun, B. (2020). Predictive multiplicity in classification. *ICML*.
- Dong, J. & Rudin, C. (2020). Exploring the cloud of variable importance for the set of all good models. *Nature Machine Intelligence*, 2, 810--824.
- Semenova, L., Rudin, C., & Parr, R. (2022). On the existence of simpler machine learning models. *FAccT*.
