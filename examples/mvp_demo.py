"""
StableGLM / rashomon-py MVP Demo
=================================

Demonstrates core functionality:
- GLM fitting with epsilon calibration
- Rashomon set sampling (ellipsoid and Hit-and-Run)
- Diagnostics (ESS/min, chords, isotropy)
- Variable Importance Clouds (VIC)
- Model Class Reliance (MCR) with correlation-aware permutations
"""
import numpy as np
from rashomon import RashomonSet

# Set random seed for reproducibility
np.random.seed(42)

print("=" * 70)
print("StableGLM MVP Demo - Rashomon Set Analysis for GLMs")
print("=" * 70)

# Generate synthetic dataset
print("\n1. Generating synthetic logistic regression data...")
n, d = 100, 5
rng = np.random.default_rng(42)
X = rng.normal(size=(n, d))
true_weights = np.array([2.0, -1.5, 0.8, 0.0, -0.5])
logits = X @ true_weights
p = 1.0 / (1.0 + np.exp(-logits))
y = (rng.random(n) < p).astype(float)

print(f"   Dataset: n={n}, d={d}")
print(f"   True weights: {true_weights}")
print(f"   Class balance: {np.mean(y):.2%}")

# Fit Rashomon set
print("\n2. Fitting Rashomon set with epsilon calibration...")
rs = RashomonSet(
    estimator="logistic",
    C=1.0,
    epsilon=0.05,
    epsilon_mode="percent_loss",
    sampler="ellipsoid",
    measure="lr",
    random_state=42
)
rs.fit(X, y)

print(f"   [OK] Model fitted successfully")
print(f"   Optimal loss L_hat: {rs.diagnostics()['L_hat']:.6f}")
print(f"   Epsilon: {rs.diagnostics()['epsilon']:.6f}")
print(f"   theta_hat norm: {rs.diagnostics()['theta_norm_2']:.4f}")

# Display diagnostics
print("\n3. Model diagnostics:")
diag = rs.diagnostics()
print(f"   Condition number kappa(H): {diag['kappa_H_est']:.2e}")
print(f"   Min signed margin: {diag['min_signed_margin']:.4f}")
print(f"   BLAS vendor: {diag['blas_vendor']}")

# Coefficient intervals (ellipsoid bounds)
print("\n4. Computing coefficient intervals (ellipsoid certificate)...")
intervals = rs.coef_intervals()
print(f"   Feature | theta_hat | [min,  max]         | True")
print(f"   " + "-" * 55)
for j in range(d):
    print(f"   {j:7d} | {rs.coef_[j]:8.4f} | [{intervals[j,0]:6.3f}, {intervals[j,1]:6.3f}] | {true_weights[j]:6.3f}")

# Sampling from Rashomon set
print("\n5. Sampling from Rashomon set (ellipsoid)...")
samples_ell = rs.sample_ellipsoid(n_samples=200, random_state=123)
print(f"   [OK] Generated {samples_ell.shape[0]} samples")

# Compute sample diagnostics
diag_samples = rs.compute_sample_diagnostics(samples_ell)
print(f"   Set fidelity: {diag_samples['set_fidelity']:.2%}")
print(f"   Chord mean: {diag_samples['chord_mean']:.4f}")
print(f"   Chord std: {diag_samples['chord_std']:.4f}")
print(f"   Isotropy ratio: {diag_samples['isotropy_ratio']:.2f}")

# Hit-and-Run sampling
print("\n6. Sampling with Hit-and-Run (with diagnostics)...")
samples_har = rs.sample_hitandrun(
    n_samples=150,
    burnin=100,
    thin=2,
    random_state=456,
    compute_diagnostics=True
)
print(f"   [OK] Generated {samples_har.shape[0]} samples")
har_diag = rs.diagnostics()["sampler_diagnostics"]
print(f"   Set fidelity: {har_diag['set_fidelity']:.2%}")
print(f"   Isotropy ratio: {har_diag['isotropy_ratio']:.2f}")
if har_diag['ess_per_param'] is not None:
    print(f"   ESS/param (mean): {np.mean(har_diag['ess_per_param']):.1f}")

# Variable Importance Cloud (VIC)
print("\n7. Computing Variable Importance Cloud (VIC)...")
vic = rs.variable_importance_cloud(
    n_samples=200,
    feature_names=[f"X{i}" for i in range(d)],
    random_state=789
)
print(f"   [OK] VIC computed with {vic['samples'].shape[0]} samples")
print(f"\n   Feature | Mean    | Std     | [5%,  95%]")
print(f"   " + "-" * 45)
for j in range(d):
    print(f"   X{j}     | {vic['mean'][j]:7.4f} | {vic['std'][j]:7.4f} | "
          f"[{vic['quantiles'][0.05][j]:6.3f}, {vic['quantiles'][0.95][j]:6.3f}]")

# Model Class Reliance (MCR)
print("\n8. Computing Model Class Reliance (MCR)...")
mcr = rs.model_class_reliance(
    X, y,
    n_permutations=10,
    n_samples=50,
    perm_mode="iid",
    random_state=321
)
print(f"   [OK] MCR computed (perm_mode='iid')")
print(f"   Base score: {mcr['base_score']:.4f}")
print(f"\n   Feature | Importance | Std")
print(f"   " + "-" * 35)
for j in range(d):
    print(f"   X{j}     | {mcr['feature_importance'][j]:10.6f} | {mcr['importance_std'][j]:7.6f}")

# MCR with correlation-aware mode
print("\n9. MCR with residual permutation (correlation-aware)...")
mcr_res = rs.model_class_reliance(
    X, y,
    n_permutations=10,
    n_samples=50,
    perm_mode="residual",
    random_state=654
)
print(f"   [OK] MCR computed (perm_mode='residual')")
print(f"\n   Feature | IID         | Residual")
print(f"   " + "-" * 40)
for j in range(d):
    print(f"   X{j}     | {mcr['feature_importance'][j]:11.6f} | {mcr_res['feature_importance'][j]:11.6f}")

# Prediction with uncertainty
print("\n10. Predictions with uncertainty (probability bands)...")
X_test = rng.normal(size=(5, d))
pbands = rs.probability_bands(X_test)
preds = rs.predict_proba(X_test)
print(f"   Sample | p_hat(y=1) | [p_min, p_max]")
print(f"   " + "-" * 40)
for i in range(5):
    print(f"   {i:6d} | {preds[i,1]:.4f} | [{pbands[i,0]:.4f}, {pbands[i,1]:.4f}]")

# Summary
print("\n" + "=" * 70)
print("MVP Demo Complete!")
print("=" * 70)
print("\nKey features demonstrated:")
print("  [+] GLM fitting with epsilon-Rashomon set calibration")
print("  [+] Ellipsoid and Hit-and-Run sampling")
print("  [+] Comprehensive diagnostics (ESS, chords, isotropy)")
print("  [+] Variable Importance Clouds (VIC)")
print("  [+] Enhanced Model Class Reliance (MCR)")
print("  [+] Prediction intervals and uncertainty quantification")
print("\nNext steps:")
print("  - Try rs.plot_vic() for visualization")
print("  - Experiment with different epsilon values")
print("  - Use Hit-and-Run for better exploration of complex sets")
print("  - Apply to real datasets!")

