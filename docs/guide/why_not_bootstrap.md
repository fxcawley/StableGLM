# Why Not Just Bootstrap?

If you want to know whether your model's conclusions are stable, bootstrap
confidence intervals are the obvious first tool. This page explains what
Rashomon analysis adds and when it matters.

## They measure different things

**Bootstrap CIs** measure sampling uncertainty: "how much would theta-hat
change if we redrew the data?" Width shrinks as O(1/sqrt(n)).

**Rashomon intervals** measure model multiplicity: "how many different
parameter vectors achieve loss within epsilon of the optimum on this
data?" Width depends on epsilon and the loss surface geometry, not on
sample size directly.

These are not two estimates of the same quantity. They answer different
questions about different sources of variation.

## The empirical gap

On the Breast Cancer dataset (d=10, n=569), Rashomon intervals are
4-11x wider than 90% bootstrap CIs. This is not a calibration error.
It reflects that many different parameter vectors fit the data nearly
as well, a fact bootstrap does not capture.

The width ratio depends on epsilon and n. Bootstrap CIs shrink with
more data; Rashomon intervals for fixed epsilon do not. The gap grows
with sample size.

## Adding the Bayesian baseline

Under a Laplace approximation with Gaussian prior, the ordering across
all features is:

    Bootstrap << Rashomon << Bayesian

- **Bootstrap** (narrowest): sampling fluctuation of the MLE.
- **Rashomon** (middle): the epsilon-sublevel set of the loss.
- **Bayesian** (widest): full posterior uncertainty including the prior.

At the typical 3% loss tolerance, the Rashomon set is a subset of the
Bayesian posterior. All Rashomon-flagged instability is already captured
by Bayesian uncertainty, but the Rashomon framing is *operational* (about
competing models) rather than *epistemic* (about beliefs).

## When bootstrap is sufficient

- You only care about sampling uncertainty (will the coefficient be
  significant with more data?).
- Model multiplicity is not a concern in your domain.
- You are comfortable with single-model reasoning and do not need to
  demonstrate robustness across near-optimal alternatives.

## When you need Rashomon analysis

- Individual decisions matter (clinical, legal, financial).
- You need to demonstrate that conclusions do not depend on which
  near-optimal model was selected.
- Correlated features create substitutability that bootstrap does not
  capture. Bootstrap CIs may be tight for each feature individually
  while the Rashomon set shows that different feature weightings achieve
  the same loss.
- A regulator or auditor asks: "would a different equally-good model
  give a different answer for this person?"

## Code example

```python
comp = rs.compare_to_bootstrap(
    X, y,
    n_bootstrap=500, n_rashomon=500,
    confidence=0.90,
    feature_names=feature_names,
    random_state=42,
)

for row in comp['comparison']:
    print(f"{row['feature']}: bootstrap={row['bootstrap_width']:.3f}, "
          f"rashomon={row['rashomon_width']:.3f}, ratio={row['width_ratio']:.1f}x")
```

A width ratio of 1x means bootstrap and Rashomon agree (rare). A ratio of
5-10x means the Rashomon set contains models qualitatively more diverse than
sampling variation would suggest. That is the finding worth reporting.
