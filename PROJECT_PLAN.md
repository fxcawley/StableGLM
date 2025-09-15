# StableGLM Project Plan

## Overview

Goal: Deliver `rashomon-py`, a rigorously grounded toolkit to define and sample the ε-Rashomon set for convex ERM GLMs, expose set-level interpretability objects (VIC, Shapley-VIC, MCR), and compute predictive multiplicity metrics (ambiguity, discrepancy, capacity). See `prop.md` for scientific details.

## Milestones (Gates)

- Gate 1 (end W2): GLMs + ε calibration (incl. bootstrap fallback) + ellipsoid certificates validated.
- Gate 2 (end W5): Hit-and-Run validated; if ESS/min remains poor on >2 datasets despite preconditioning, trigger kill-switch.
- Gate 3 (end W7): VIC/MCR/plots functional; Shapley-VIC runtime guardrails in place.
- Gate 4 (end W9): Ambiguity/discrepancy/capacity feature-complete; thresholds (T1,T2,T3) wired.
- Gate 5 (end W11): Benchmarks, baselines, and red-team experiments complete.
- Gate 6 (W12): PyPI release and paper-ready report.

## Priorities (P0–P2)

- P0: Core correctness & optics — ε calibration with Wilks checks + bootstrap fallback; separation/conditioning guardrails; ellipsoid certificates; robust sampler core; diagnostics.
- P1: Metrics fidelity & UX — VIC, MCR (with correlation-aware permutations), Shapley-VIC with runtime caps, plots, dataset loaders, scaling/tightness studies.
- P2: Nice-to-haves — experimental trust-region surrogate for MCR, expanded examples, integrations.

## Work Packages (WPs)

- WP-A: Infrastructure & Docs
  - Repo scaffold, CI matrix (OpenBLAS/MKL), coverage, docs (Sphinx+MyST), governance, .gitignore.
- WP-B: GLM Core & Calibration
  - Deterministic solvers (L2), HVPs, CG norms, ε calibration (percent, LR-α low-dim), penalized bootstrap fallback; global `measure ∈ {param, lr}`.
- WP-C: Certificates & Validity
  - Linear functional hacking intervals, margin/prediction bands (Lipschitz + tight 1D), conservative trust-region validity radius with plot shading.
- WP-D: Exact Sampling
  - Membership oracle, Hit-and-Run with robust line search, optional LR-geometry whitening, H^{-1/2} approx via Lanczos/Chebyshev with fallback; diagnostics (ESS/min, chords, isotropy, set-fidelity); validation suite.
- WP-E: Metrics Layer
  - VIC, Shapley-VIC (stratified sampling, CI early stop), MCR with `perm = {iid, group, residual}`, Discrepancy (deterministic bound + empirical pairs), Rashomon capacity (δ-covering, experimental if needed), threshold regimes T1–T3.
- WP-F: Evaluation & Fairness
  - Datasets (UCI, credit, healthcare, COMPAS), fairness toggles and ethics/policy notes, tightness/scaling studies, red-team experiments (collinearity sweep, proxy fairness toy), baselines.
- WP-G: Release & Paper
  - API hardening, plotting API, tutorials, packaging & PyPI, paper-style report & MLOSS draft.

## Schedule (12 weeks)

- W1–W2: WP-B (core + calibration), WP-C (certificates), WP-A partial.
- W3–W5: WP-D (sampler), WP-C validity radius.
- W6–W7: WP-E (VIC/MCR/Shapley-VIC) + WP-G plotting.
- W8–W9: WP-E (ambiguity, discrepancy, capacity), WP-F fairness & loaders.
- W10–W11: WP-F (benchmarks, baselines, red-team), WP-G tutorials.
- W12: WP-G release & paper.

## Acceptance Criteria (high-level)

- Numerical: ≥90% unit coverage for core math; CG/Lanczos/H–R tolerance tests; κ(H) and W diagnostics enforced.
- Calibration: α printed for any ε; Wilks preconditions checked; bootstrap fallback operational within 60s on n ≤ 1e4.
- Sampler: Isotropy checked; diagnostics (ESS/min, chords, trust fraction) reported; kill-switch policy documented.
- Fairness: Threshold regimes explicit; report (T1) and (T2) by default; ethics note present when sensitive datasets used.
- Reproducibility: Seeds, software stack, BLAS vendor logged; end-to-end examples < 3 minutes CPU.

## Risks & Mitigations

- Poor mixing on ill-conditioned problems — preconditioning, ε auto-shrink, ellipsoid-only fallback with set-fidelity reporting (Gate 2 kill-switch).
- Wilks misinterpretation in penalized/high-dim — explicit checks; bootstrap fallback with CI.
- Permutation MCR instability under correlation — conditional/residual permutations; warnings on collinearity.
- Reviewer optics (measure, fairness thresholds) — global measure flag; thresholds T1–T3 explicit in API and plots.

## Governance & Roles

- Tech Lead (Research Architect): owns scientific validity, API and acceptance criteria.
- Core Engineer: owns implementations for GLM/HVP/CG/sampler.
- Metrics Engineer: owns VIC/MCR/capacity implementations and plots.
- Docs/Release: owns docs, tutorials, packaging, and paper draft.

## Issue Backlog (initial)

Format: ID | Title | Labels | Est | Deps | Sprint

- A1 | Repo scaffold and dev tooling | [infra,P0] | 1d | — | W1
- A2 | CI matrix, wheels, coverage, BLAS diversity | [infra,P0] | 1d | A1 | W1
- A3 | Docs scaffold (Sphinx+MyST), API autodoc | [docs,P1] | 1d | A1 | W1
- A4 | Governance and licensing | [ops,P2] | 0.5d | — | W1
- B5 | `RashomonSet` API skeleton | [core,P0] | 1d | A1 | W1
- B5b | Global measure setting and annotation | [core,ux,P0] | 0.5d | B5 | W1
- B6 | Deterministic GLM solvers + guardrails | [core,P0] | 2d | B5 | W1–W2
- B7 | HVP API | [core,P0] | 1d | B6 | W2
- B8 | CG solves + conditioning diagnostics | [core,P0] | 2d | B7 | W2
- B9 | ε calibration (Wilks checks) | [core,P0] | 1d | B6 | W1
- B9b | Penalized LR bootstrap fallback | [core,stats,P0] | 1d | B6,B9 | W1
- C10 | Linear hacking intervals | [bounds,P0] | 1d | B8,B9 | W2
- C11 | Per-point margin/p-bands (Lipschitz) | [bounds,P0] | 1.5d | C10 | W2
- C12 | Tight p-intervals (1D) | [bounds,P1] | 2d | C11 | W2–W3
- C13 | Trust-region validity radius | [bounds,P1] | 1d | B7–B9 | W2–W3
- D14 | Membership oracle | [sampling,P0] | 1d | B6–B9,C10–C13 | W3
- D15 | Hit-and-Run with robust line search | [sampling,P0] | 2d | D14 | W3
- D16 | Directions + preconditioning toggle | [sampling,P1] | 1d | D15 | W3
- D17 | H^{-1/2} approx (Lanczos/Chebyshev) | [sampling,P1] | 2d | B8,D16 | W4
- D18 | Diagnostics (ESS/min, chords, isotropy) | [sampling,ux,P0] | 1d | D15–D17 | W4
- D19 | Sampler validation suite | [sampling,P1] | 1d | D18 | W4–W5
- E20 | VIC + plots | [metrics,P0] | 2d | D18–D19 | W6
- E21 | Shapley-VIC (runtime/quality caps) | [metrics,P1] | 2d | E20 | W6–W7
- E22 | MCR (perm) with correlation options | [metrics,P0] | 2d | D18,E20 | W6–W7
- E23 | MCR trust-region surrogate (exp) | [metrics,experimental,P3] | 2d | E22 | W7
- E24 | Ambiguity via margins | [metrics,P0] | 1d | C11–C12 | W8
- E24b | Threshold regimes API | [metrics,ux,P0] | 0.5d | E24 | W2 (pulled up)
- E25 | Discrepancy bound + pairs | [metrics,P0] | 2d | D18,E24 | W8–W9
- E26 | Rashomon capacity (δ-covering) | [metrics,stats,P1] | 3d | D18 | W8–W9
- F27 | Datasets + fairness toggles/policy | [data,fairness,P0] | 2d | E24–E26 | W8–W9
- F28 | Tightness study | [eval,P0] | 2d | C10–C13,D19 | W9–W10
- F29 | Scaling study | [eval,P1] | 2d | D18–D19 | W10–W11
- F30 | Baselines (CIs, bootstrap VI, L1, NN) | [eval,P1] | 3d | E20–E26 | W10–W11
- F31 | Red-team: collinearity sweep | [eval,redteam,P1] | 1d | E20–E26 | W10
- F32 | Red-team: proxy fairness toy | [fairness,redteam,P1] | 1d | F27,E24b | W10
- G31 | API hardening + diagnostics | [ux,P0] | 1d | D18,B9–B9b,C13 | W10
- G32 | Plotting API | [ux,plots,P1] | 1.5d | E20–E26 | W10
- G33 | Tutorials and examples | [docs,P1] | 2d | G32,F28–F30 | W11
- G34 | Packaging and PyPI release | [release,P0] | 1d | A2–A3,G31–G33 | W12
- G35 | Paper-style report + MLOSS draft | [paper,P1] | 3d | F28–F32,G33 | W12
- G36 | `from_ensemble(models)` parity | [api,P1] | 1.5d | E20–E26 | W7–W8

## References

- See `prop.md` for formal definitions and the API sketch.
