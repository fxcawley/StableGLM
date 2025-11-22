import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from rashomon import RashomonSet

def verify_loss_bounds():
    print("Verifying Rashomon Set Loss Bounds...")
    
    # 1. Setup Data (Same as tutorial)
    data = load_breast_cancer()
    X = data.data[:, [0, 1, 4, 3, 6]] 
    y = data.target
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 2. Fit RashomonSet (epsilon=0.01)
    epsilon = 0.01
    rs = RashomonSet(
        estimator="logistic",
        epsilon=epsilon, 
        epsilon_mode="percent_loss",
        sampler="hitandrun",
        random_state=42,
        safety_override=True,
        C=0.5
    )
    rs.fit(X_train, y_train)
    
    # 3. Get Optimal Loss and Threshold
    L_hat = rs.diagnostics()["L_hat"]
    threshold_loss = L_hat * (1 + epsilon)
    print(f"Optimal Loss (L_hat): {L_hat:.6f}")
    print(f"Allowed Loss (Threshold): {threshold_loss:.6f}")
    
    # 4. Sample models
    n_samples = 1000
    samples = rs.sample(n_samples=n_samples)
    
    # 5. Calculate Loss for EACH sample
    # Manual log-loss calculation to be absolutely sure
    losses = []
    for theta in samples:
        logits = X_train @ theta
        # Sigmoid with clipping for stability
        p = 1 / (1 + np.exp(-logits))
        p = np.clip(p, 1e-15, 1 - 1e-15)
        loss = -np.mean(y_train * np.log(p) + (1 - y_train) * np.log(1 - p))
        # Add regularization term (since L_hat includes it?)
        # Wait, RashomonSet L_hat usually includes regularization if optimized with it.
        # Let's check rs.score vs internal loss.
        # The objective function minimized includes ridge penalty: L(theta) + lambda * ||theta||^2
        # The epsilon constraint is on THIS objective.
        # So I must include regularization in the check.
        reg_term = 0.5 * (1.0 / rs.C) * np.sum(theta**2)
        total_obj = loss + reg_term
        losses.append(total_obj)
        
    losses = np.array(losses)
    max_sample_loss = np.max(losses)
    
    print(f"Max Sample Loss: {max_sample_loss:.6f}")
    
    if max_sample_loss > threshold_loss + 1e-5:
        print(f"FAIL: Samples violate bounds! Diff: {max_sample_loss - threshold_loss:.6f}")
    else:
        print("PASS: All samples are within the epsilon-Rashomon set.")
        
    # 6. Check 'mean area' variation specifically
    # Index 3 is mean area
    area_coeffs = samples[:, 3]
    print(f"\nMean Area Coefficient Stats:")
    print(f"Optimal: {rs._theta_hat[3]:.4f}")
    print(f"Range: [{np.min(area_coeffs):.4f}, {np.max(area_coeffs):.4f}]")
    print(f"Std Dev: {np.std(area_coeffs):.4f}")

if __name__ == "__main__":
    verify_loss_bounds()

