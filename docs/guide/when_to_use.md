# When This Package Helps

## Use rashomon-py when

**You trained a logistic or linear regression and need to know if conclusions are stable.**
Regulatory audits, clinical decision support, and credit scoring all have situations
where individual predictions matter. If someone asks "would a different equally-good
model give this patient a different diagnosis?", rashomon-py answers that directly.

**You want to compare model multiplicity to bootstrap uncertainty.**
Bootstrap CIs are tight, your model looks solid, but you suspect correlated features
create multiple valid explanations. Rashomon intervals are typically 4-11x wider than
bootstrap CIs on the same data -- they capture a different axis of uncertainty.

**You need to flag unstable individual predictions.**
Ambiguity analysis identifies exactly which instances flip across near-optimal models.
You get a list of patients, applicants, or cases that deserve additional scrutiny.

**You want to understand feature substitutability.**
Correlated features often have wide coefficient distributions across the Rashomon set,
meaning the data does not determine how much weight to assign each one. The VIC
(coefficient distribution) and MCR (min/max importance) outputs expose this directly.

## Do not use rashomon-py when

**Your model is not L2-regularized logistic or linear regression.**
No trees, no neural nets, no SVMs, no L1 or elastic-net penalties. The mathematics
are specific to the L2-regularized convex loss surface.

**You want fairness metrics or bias auditing.**
rashomon-py measures prediction instability, not group-level disparities. It can
show that predictions are unstable for a subgroup, but it does not compute fairness
metrics like demographic parity or equalized odds.

**You want model selection.**
This tool audits a model you already trained. It tells you whether the conclusions
from that model are robust. It does not pick a better model for you.

**Your feature space is very high-dimensional (d > ~50).**
Certificate-based estimates become conservative at high d (7-20x overestimates at
d=60+). Hit-and-Run MCMC mixing degrades. Consider PCA or feature selection to
reduce dimensionality before auditing, or use certificates as conservative upper
bounds and interpret accordingly.

**You want p-values or hypothesis tests.**
Rashomon analysis is about model multiplicity (how many different parameter vectors
achieve near-optimal loss), not statistical inference (whether a coefficient is
significantly different from zero). These are different questions.
Bootstrap CIs answer the inference question. Rashomon intervals answer the
multiplicity question. See {doc}`why_not_bootstrap` for the full comparison.
