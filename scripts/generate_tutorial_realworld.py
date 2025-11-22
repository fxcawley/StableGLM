import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import ssl
from sklearn.datasets import fetch_openml

# Fix SSL certificate errors
ssl._create_default_https_context = ssl._create_unverified_context

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from rashomon import RashomonSet

def generate_images():
    output_dir = os.path.join("docs", "_static")
    os.makedirs(output_dir, exist_ok=True)
    
    print("Fetching 'German Credit' dataset (ID 31)...")
    try:
        data = fetch_openml(data_id=31, as_frame=True, parser="auto")
    except Exception as e:
        print(f"Failed to fetch data: {e}")
        return

    X_full = data.data
    # German Credit target is 'good'/'bad'. Map to 0/1.
    y_full = (data.target == 'bad').astype(float) # 1 = Bad credit (Default)

    # Check available columns
    print("Columns:", X_full.columns.tolist())
    
    # German Credit has meaningful names.
    # We'll use a subset for clarity.
    # 'checking_status', 'duration', 'credit_history', 'purpose', 'credit_amount', 'savings_status', 'employment', 'age'
    
    selected_features = ['checking_status', 'duration', 'credit_history', 'credit_amount', 'age', 'housing', 'job']
    cols = [c for c in selected_features if c in X_full.columns]
    
    # Fallback if names don't match
    if not cols:
        print("Warning: Column names not found. Using first 10 columns.")
        X = X_full.iloc[:, :10].copy()
        cols = X.columns.tolist()
    else:
        X = X_full[cols].copy()
        
    # Preprocessing
    # Determine categorical/numeric automatically
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    print(f"Numeric: {numeric_features}")
    print(f"Categorical: {categorical_features}")

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='if_binary', sparse_output=False, handle_unknown='ignore'), categorical_features)
        ],
        verbose_feature_names_out=False
    )
    
    # Fit pipeline on a subset to get names
    X_processed = preprocessor.fit_transform(X)
    feature_names = preprocessor.get_feature_names_out()
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y_full, test_size=0.2, random_state=42)
    
    # Subsample NOT needed for German Credit (n=1000)
    
    print(f"Training RashomonSet on n={X_train.shape[0]}, d={X_train.shape[1]}...")
    
    # Fit RashomonSet
    rs = RashomonSet(
        estimator="logistic",
        epsilon=0.05, 
        epsilon_mode="percent_loss", # 5% worse than optimal
        sampler="hitandrun",
        random_state=42
    )
    rs.fit(X_train, y_train)
    
    print(f"Train Score (Accuracy): {rs.score(X_train, y_train):.4f}")
    
    # 1. VIC Plot
    print("Generating VIC plot...")
    plt.figure(figsize=(10, 6))
    # Use plotting helper directly
    # We need to map feature names if possible. 
    # plot_vic doesn't take feature names yet? It uses indices.
    # But we can pass a list of names to `feature_names` argument if supported?
    # Checking plotting.py... plot_vic signature: (vic_result, theta_hat=None, feature_names=None, ...)
    # RashomonSet.plot_vic delegates to it.
    
    rs.plot_vic(feature_names=feature_names)
    plt.title("Variable Importance Cloud: Credit Default Risk")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "tutorial_vic.png"))
    plt.close()
    
    # 2. Ambiguity Plot
    print("Generating Ambiguity plot...")
    plt.figure(figsize=(10, 6))
    # Check ambiguity on test set (first 30)
    rs.plot_ambiguity(X_test[:30], y=y_test[:30])
    plt.title("Predictive Multiplicity: Credit Default (Test Set)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "tutorial_ambiguity.png"))
    plt.close()
    
    # 3. MCR (Print to stdout, I'll copy to markdown)
    print("Computing MCR...")
    mcr = rs.model_class_reliance(X_train, y_train, perm_mode="residual", n_permutations=10)
    
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
    print("\nMCR Table:")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(mcr_df)

if __name__ == "__main__":
    generate_images()

