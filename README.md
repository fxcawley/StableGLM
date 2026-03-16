# rashomon-py

**Audit whether a linear or logistic model's conclusions are stable across equally good alternatives.**

You trained a logistic regression. It scores well. But would a different
equally-good model give the same predictions? The same feature importances?
The same decisions for individual patients?

rashomon-py answers these questions for L2-regularized logistic and linear
regression by characterizing the set of all near-optimal models (the
epsilon-Rashomon set) and measuring what changes across it.

## Who this is for

- sklearn users working with tabular data who care about interpretability
- Anyone who wants to know whether their model's explanations are an artifact
  of one arbitrary fit or a robust property of the data

## Install

```bash
pip install -e .
```

## Quickstart

```python
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from rashomon import RashomonSet

X, y = load_breast_cancer(return_X_y=True)
X = StandardScaler().fit_transform(X[:, :10])

rs = RashomonSet(estimator="logistic", epsilon=0.03,
                 epsilon_mode="percent_loss", random_state=0).fit(X, y.astype(float))

# How many patients get a different diagnosis under an equally-good model?
amb = rs.ambiguity(X, threshold_mode="fixed", threshold_value=0.5)
print(f"Ambiguity: {amb['ambiguity_rate']:.1%}")

# How much do coefficients vary across the Rashomon set?
vic = rs.variable_importance_cloud(n_samples=200)
print(f"Coefficient std: {vic['std']}")
```

## What the outputs mean

| Output | What it tells you |
|--------|-------------------|
| **Ambiguity rate** | Fraction of instances where some near-optimal model flips the predicted label. |
| **Discrepancy** | Worst-case disagreement rate between any two models in the set. |
| **Coefficient spread (VIC)** | Distribution of each coefficient across near-optimal models. Wide spread = the feature's role is not pinned down by the data. |
| **Probability bands** | Per-instance [min, max] predicted probability across the Rashomon set. |
| **MCR bounds** | Min/max feature importance (Model Class Reliance) across the set. |

Two computation modes:

- **Certificates** (ellipsoidal, closed-form) -- fast upper bounds, milliseconds.
  Tight at low dimensionality (within 1.3x at d=10).
- **Hit-and-Run MCMC** -- slower, asymptotically exact membership sampling from
  the true Rashomon set. Requires adequate chain length for mixing (see
  limitations).

## Benchmark summary

Results on real datasets with CV-selected regularization. Full details in
[docs/evaluation.md](docs/evaluation.md).

| Dataset | d | Cert. Ambiguity | Exact Ambiguity | Cert/Exact | Fit time |
|:--------|--:|:---------------:|:---------------:|:----------:|:--------:|
| Breast Cancer PCA-10 | 10 | 36.0% | 27.0% | 1.3x | 0.02s |
| Breast Cancer Full | 30 | 18.4% | 10.0% | 1.8x | 0.02s |
| German Credit | 61 | 85.4% | 11.4% | 7.5x | 0.05s |
| Adult Census | 104 | 79.0% | 9.2% | 8.6x | 0.11s |

At d < 20, certificates are a reliable fast screen. At d > 60, they are
conservative upper bounds -- use Hit-and-Run for precise estimates.

## Limitations

- **Only L2-regularized linear and logistic regression.** No trees, no neural nets,
  no L1 penalties.
- **Certificate estimates are upper bounds** and grow conservative in higher
  dimensions (see benchmark table above).
- **Hit-and-Run sampling gets harder in high dimensions.** At d=104, effective sample
  size is low even after 500 draws. Long chains or dimensionality reduction help.
- **Results depend on epsilon.** The Rashomon set is bigger when you allow more loss
  tolerance. The `epsilon_mode="percent_loss"` default is a reasonable starting point
  but users should run sensitivity analysis (the tutorial shows how).

## Scope

rashomon-py supports **L2-regularized logistic and linear regression only**.
This is a deliberate constraint, not a roadmap gap.

Out of scope:
- Tree models, neural nets, or arbitrary sklearn estimators
- L1, elastic-net, or other penalty types
- Fairness guarantees or bias auditing
- Model selection (this tool audits a model you already trained)

## Documentation

- [Quickstart](docs/guide/quickstart.md) -- Get running in 20 lines
- [When to use this](docs/guide/when_to_use.md) -- And when not to
- [Choosing epsilon](docs/guide/choosing_epsilon.md) -- The fundamental parameter
- [Certificates vs sampling](docs/guide/certificates_vs_sampling.md) -- Which mode to use
- [Interpreting instability](docs/guide/interpreting_instability.md) -- What the outputs mean in practice
- [Why not bootstrap?](docs/guide/why_not_bootstrap.md) -- What Rashomon adds beyond CIs
- [Tutorial](docs/examples/tutorial.md) -- Full case study: "When Equally-Good Models Disagree"
- [Evaluation](docs/evaluation.md) -- Benchmark results on 4 real datasets
- [API Reference](docs/api/reference.rst) -- Complete API docs

## Key references

- Fisher, Rudin, Dominici (2019). "All Models are Wrong, but Many are Useful." *JMLR*. -- MCR framework
- Marx, Calmon, Ustun (2020). "Predictive Multiplicity in Classification." *ICML*. -- Ambiguity/discrepancy metrics
- Dong & Rudin (2020). "Exploring the Cloud of Variable Importance." *Nature Machine Intelligence*. -- VIC concept

## License

MIT
