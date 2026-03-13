# Overview

## What StableGLM Does

StableGLM characterizes the **Rashomon set** for regularized linear and logistic regression:
the set of all parameter vectors that achieve loss within $\epsilon$ of the optimum. It provides:

**Certificates** (closed-form, fast):
- **Hacking intervals**: the range of any linear functional $s^\top\theta$ across the Rashomon set
- **Coefficient intervals**: per-feature parameter ranges
- **Probability bands**: per-instance prediction ranges
- **Ambiguity**: fraction of instances where the Rashomon set contains models that disagree on the label

**Sampling** (exact, slower):
- **Ellipsoid sampling**: fast but approximate (quadratic approximation of the loss)
- **Hit-and-Run**: exact membership sampling from the true Rashomon set via MCMC

**Analysis**:
- **VIC** (Variable Importance Cloud): distribution of coefficients across the Rashomon set
- **MCR** (Model Class Reliance): min/max feature importance bounds (Fisher et al. 2019)
- **Discrepancy**: worst-case disagreement rate between any two models in the set
- **Bootstrap comparison**: side-by-side comparison of bootstrap CIs vs. Rashomon intervals

## Why This Matters

Standard model evaluation reports a single fit and quantifies *sampling uncertainty* (bootstrap CIs,
p-values). This answers: "how uncertain are we about the best-fit parameters?"

Rashomon set analysis answers a different question: "how many qualitatively different models
perform nearly as well?" These two quantities can diverge by orders of magnitude. A model
can have narrow CIs (well-identified parameters) while the Rashomon set contains models with
opposite coefficient signs (massive design-choice multiplicity).

The {doc}`case study <../examples/tutorial>` demonstrates a 85-230x width ratio between
bootstrap CIs and Rashomon intervals on a standard medical dataset.

## Supported Models

| Model | Loss | Regularization |
|-------|------|----------------|
| Logistic regression | $\frac{1}{n}\sum[\log(1+e^z) - yz] + \frac{\lambda}{2}\|\theta\|^2$ | L2 |
| Ridge regression | $\frac{1}{2n}\sum(y-z)^2 + \frac{\lambda}{2}\|\theta\|^2$ | L2 |

## Links

- {doc}`quickstart`: Get running in 20 lines
- {doc}`../examples/tutorial`: Full case study with real data
- {doc}`../api/reference`: API documentation
- {doc}`../reproducibility`: Reproducibility notes
