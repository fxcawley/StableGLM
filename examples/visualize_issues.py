"""Demonstrate visualization of predictive multiplicity on Adult dataset."""

import os
import numpy as np
from sklearn.model_selection import train_test_split

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib not available, skipping visualization demo")
    exit(0)

from rashomon import RashomonSet
from tests.test_adult_dataset import load_adult_data

def main():
    print("Loading Adult dataset...")
    data_path = os.path.join("tests", "data", "adult.data")
    if not os.path.exists(data_path):
        print("Please run scripts/download_data.py first")
        return

    X, y = load_adult_data(data_path, n_samples=2000)
    
    # Split for clean evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    print(f"Fitting Rashomon set (n_train={len(X_train)})...")
    rs = RashomonSet(
        estimator="logistic",
        C=0.01,
        epsilon_mode="percent_loss",
        epsilon=0.05,
        random_state=42
    ).fit(X_train, y_train)
    
    print("Generating visualizations...")
    
    # 1. Plot VIC (top 10 features)
    # Need feature names for clarity - we'll just use indices as we lost names in helper
    feature_names = [f"Feature {i}" for i in range(X.shape[1])]
    
    # Find top variable features (highest variance in Rashomon set)
    vic_res = rs.variable_importance_cloud(n_samples=50, sampler="ellipsoid")
    top_vars_idx = np.argsort(vic_res["std"])[-10:]
    
    # Filter vic_res to top 10 for plotting
    vic_res_top = {
        "samples": vic_res["samples"][:, top_vars_idx],
        "mean": vic_res["mean"][top_vars_idx],
        "intervals": vic_res["intervals"][top_vars_idx],
        "feature_names": [feature_names[i] for i in top_vars_idx]
    }
    
    # Pass show_theta_hat=False because we are plotting a subset of features
    # and the wrapper would try to plot the full theta_hat
    fig_vic, _ = rs.plot_vic(vic_result=vic_res_top, show_theta_hat=False)
    fig_vic.savefig("adult_vic.png")
    print("Saved adult_vic.png")
    
    # 2. Plot Ambiguity
    print("Computing ambiguity...")
    fig_amb, _ = rs.plot_ambiguity(
        X_test, 
        threshold_mode="match_prevalence", 
        y=y_test,
        figsize=(10, 6)
    )
    fig_amb.savefig("adult_ambiguity.png")
    print("Saved adult_ambiguity.png (Shows predictive instability)")
    
    # 3. Discrepancy Heatmap
    print("Computing discrepancy heatmap...")
    fig_disc, _ = rs.plot_discrepancy(
        X_test, 
        n_samples=50, 
        threshold_mode="match_prevalence", 
        y=y_test
    )
    fig_disc.savefig("adult_discrepancy.png")
    print("Saved adult_discrepancy.png")
    
    # 4. Fixing Discrepancies: Robust Prediction (Ensemble)
    print("\n--- Fixing Discrepancies: Robust Prediction ---")
    # Strategy: Average probability across the set (Bayesian Model Averaging approx)
    # This "fixes" the issue of arbitrary selection by using the whole set.
    
    # Re-generate samples for ensemble
    samples = rs.sample(n_samples=50)
    margins = X_test @ samples.T
    
    # Compute probability for each model
    def sigmoid(z): return 1 / (1 + np.exp(-z))
    probas_ensemble = np.mean(sigmoid(margins), axis=1)
    
    # Compute single best model (theta_hat) probability
    scores_hat = rs.decision_function(X_test)
    probas_hat = sigmoid(scores_hat)
    
    # Evaluate Log Loss (NLL)
    from sklearn.metrics import log_loss, accuracy_score
    
    loss_hat = log_loss(y_test, probas_hat)
    loss_ensemble = log_loss(y_test, probas_ensemble)
    
    print(f"Standard Model Log Loss: {loss_hat:.4f}")
    print(f"Robust Ensemble Log Loss: {loss_ensemble:.4f}")
    
    if loss_ensemble < loss_hat:
        print("SUCCESS: Ensemble reduced prediction error!")
    else:
        print("Note: Ensemble performed similarly (robustness > accuracy).")
        
    # Check Ambiguity Reduction (Hard vs Soft)
    # Soft predictions don't have "ambiguity" in the same way, they express uncertainty.
    # But we can check if the ensemble accuracy is better.
    acc_hat = accuracy_score(y_test, probas_hat > 0.5)
    acc_ensemble = accuracy_score(y_test, probas_ensemble > 0.5)
    
    print(f"Standard Accuracy: {acc_hat:.2%}")
    print(f"Ensemble Accuracy: {acc_ensemble:.2%}")

    print("\nVisualization Complete. Check the generated .png files to see the issues.")

if __name__ == "__main__":
    main()

