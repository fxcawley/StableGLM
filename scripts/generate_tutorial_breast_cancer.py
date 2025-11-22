import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from rashomon import RashomonSet

def generate_images():
    output_dir = os.path.join("docs", "_static")
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading Breast Cancer dataset...")
    data = load_breast_cancer()
    X_full = data.data
    y_full = data.target
    feature_names_full = data.feature_names
    
    # Select a subset of features for clarity
    # 'mean radius', 'mean texture', 'mean smoothness', 'mean area', 'mean concavity'
    # Radius and Area are highly correlated.
    
    selected_indices = [0, 1, 4, 3, 6] # radius, texture, smoothness, area, concavity
    X = X_full[:, selected_indices]
    feature_names = feature_names_full[selected_indices]
    
    print(f"Selected features: {feature_names}")
    
    # Scale
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y_full, test_size=0.2, random_state=42)
    
    print(f"Training RashomonSet on n={X_train.shape[0]}, d={X_train.shape[1]}...")
    
    # Fit RashomonSet
    # epsilon=0.01 (Tight set, 1% loss tolerance) to check for indispensable features
    rs = RashomonSet(
        estimator="logistic",
        epsilon=0.01, 
        epsilon_mode="percent_loss",
        sampler="hitandrun",
        random_state=42,
        safety_override=True, # Dataset is separable, need override
        C=0.5 # Stronger regularization to constrain set
    )
    rs.fit(X_train, y_train)
    
    print(f"Train Accuracy: {rs.score(X_train, y_train):.4f}")
    print(f"Optimal Loss: {rs.diagnostics()['L_hat']:.4f}")
    
    # 1. VIC Plot
    print("Generating VIC plot...")
    plt.figure(figsize=(10, 6))
    rs.plot_vic(feature_names=feature_names)
    plt.title("Variable Importance Cloud: Breast Cancer Diagnosis")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "tutorial_vic.png"))
    plt.close()
    
    # 2. Ambiguity Plot
    print("Generating Ambiguity plot...")
    plt.figure(figsize=(10, 6))
    # Use a subset of test data
    rs.plot_ambiguity(X_test[:30], y=y_test[:30])
    plt.title("Predictive Multiplicity (Test Samples)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "tutorial_ambiguity.png"))
    plt.close()
    
    # 3. MCR
    print("Computing MCR (iid)...")
    mcr = rs.model_class_reliance(X_train, y_train, perm_mode="iid", n_permutations=10)
    
    mcr_matrix = mcr["importance_matrix"]
    min_imp = np.min(mcr_matrix, axis=0)
    max_imp = np.max(mcr_matrix, axis=0)
    mean_imp = mcr["feature_importance"]
    
    mcr_df = pd.DataFrame({
        "Feature": feature_names,
        "Min Importance": min_imp,
        "Mean Importance": mean_imp,
        "Max Importance": max_imp
    })
    
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print("\nMCR Table:")
    print(mcr_df)

if __name__ == "__main__":
    generate_images()

