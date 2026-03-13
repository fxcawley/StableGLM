# StableGLM

Your model is accurate. But is it the *only* accurate model, or one of thousands?

StableGLM explores the set of all models that perform nearly as well as your best fit.
When that set is large and diverse, conclusions drawn from any single model --
feature importances, individual predictions, fairness metrics -- may be artifacts
of an arbitrary optimization path rather than properties of the data.

**What this reveals:**

- Patients whose diagnosis flips depending on which equally-good model you pick
- Features that appear important in one good model and irrelevant in another
- How much wider the space of near-optimal models is than bootstrap CIs suggest

```{toctree}
:maxdepth: 2
:caption: Contents

guide/concepts
guide/quickstart
examples/tutorial
guide/overview
api/reference
reproducibility
```
