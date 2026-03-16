# How to Interpret Instability

## Ambiguity rate

"X% of instances are ambiguous" means X% of predictions would change if you
used a different equally-good model. This is not noise or estimation error.
It means the data genuinely admits multiple valid answers for those instances.

An ambiguous patient is one whose prediction is not a property of the data
alone, but of which particular near-optimal model the practitioner happened
to fit.

## When ambiguity is high (> 20%)

High ambiguity does not mean the model is bad. Accuracy can be high and
ambiguity can still be high -- they measure different things.

**What to do:**

- Flag ambiguous instances for human review or additional data collection.
- Report prediction intervals rather than point predictions for affected
  instances.
- Consider whether the task is well-defined enough. If 30% of patients
  get different diagnoses across equally-good models, the data may not
  support confident individual predictions.
- Do NOT switch model classes on this basis alone. Ambiguity is a property
  of the data and the loss tolerance, not a deficiency of the algorithm.

## When ambiguity is zero

Conclusions are robust at this epsilon. Every instance gets the same
prediction under every near-optimal model.

**What to do:**

- Report this as a positive finding: the model's conclusions are stable.
- Consider testing a larger epsilon to find the threshold where instability
  appears. The sensitivity curve (see {doc}`choosing_epsilon`) is more
  informative than any single number.

## Coefficient spread (VIC)

**Wide distribution:** the feature's importance is not determined by the data.
Many different weightings are compatible with near-optimal loss. This often
happens with correlated features where the optimizer's particular decomposition
into coefficient weights is one of many valid solutions.

**Interval crossing zero:** some near-optimal models assign the feature
a positive coefficient and others assign it a negative one. The sign of the
feature's effect is not robust.

**Narrow distribution:** the feature's coefficient is tightly constrained
across the Rashomon set. Its role in the model is a robust property of the
data, not an artifact of the fit.

## Discrepancy

Discrepancy measures the worst-case pairwise disagreement rate between any
two models in the Rashomon set. If discrepancy is 8%, there exist two
models -- both achieving near-optimal loss -- that give opposite predictions
for 8% of instances.

This is a stronger statement than ambiguity. Ambiguity says "some model
disagrees for this instance." Discrepancy says "there is a specific pair
of models that disagree on this many instances simultaneously."

## What instability does NOT mean

**It does not mean the model is wrong.** High accuracy with high ambiguity
means the model predicts well on average but some individual predictions
are not uniquely determined by the data.

**It does not mean you should switch algorithms.** If the instability is
in the data (correlated features, overlapping classes), a different
algorithm will face the same fundamental ambiguity.

**It does not mean the features are useless.** Wide VIC intervals for
correlated features mean the data cannot distinguish which specific
combination of those features drives the prediction. The features
collectively matter; their individual contributions are underdetermined.

**It means the data permits multiple valid explanations.** The right
response is transparency: report which predictions are robust, which
are not, and let domain experts decide how to handle the ambiguous cases.
