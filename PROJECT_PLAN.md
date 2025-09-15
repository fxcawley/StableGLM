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

## Issue Tracking & Labels

- Components: [core], [sampling], [metrics], [fairness], [infra], [docs], [release], [paper].
- Priority: P0 (must), P1 (should), P2 (nice).
- Status: todo → in_progress → blocked → done.

## References

- See `prop.md` for formal definitions and the API sketch.
