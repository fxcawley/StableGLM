import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import ssl
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from rashomon import RashomonSet

# Fix SSL certificate errors
ssl._create_default_https_context = ssl._create_unverified_context

def generate_german_credit():
    output_dir = os.path.join("docs", "_static")
    os.makedirs(output_dir, exist_ok=True)
    
    print("Fetching 'German Credit' dataset (ID 31)...")
    try:
        data = fetch_openml(data_id=31, as_frame=True, parser="auto")
    except Exception as e:
        print(f"Failed to fetch data: {e}")
        return

    X_full = data.data
    y_full = (data.target == 'bad').astype(float) # 1 = Bad credit

    # Select subset
    selected_features = ['checking_status', 'duration', 'credit_history', 'credit_amount', 'age', 'housing', 'job']
    cols = [c for c in selected_features if c in X_full.columns]
    X = X_full[cols].copy()
        
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='if_binary', sparse_output=False, handle_unknown='ignore'), categorical_features)
        ],
        verbose_feature_names_out=False
    )
    
    X_processed = preprocessor.fit_transform(X)
    feature_names = preprocessor.get_feature_names_out()
    
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y_full, test_size=0.2, random_state=42)
    
    print(f"Training RashomonSet on German Credit (n={X_train.shape[0]})...")
    
    # Epsilon=0.05 (standard)
    rs = RashomonSet(
        estimator="logistic",
        epsilon=0.05, 
        epsilon_mode="percent_loss", 
        sampler="hitandrun",
        random_state=42
    )
    rs.fit(X_train, y_train)
    
    # VIC Plot
    plt.figure(figsize=(10, 6))
    rs.plot_vic(feature_names=feature_names)
    plt.title("Variable Importance Cloud: German Credit")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "german_vic.png"))
    plt.close()
    
    # MCR
    print("Computing MCR for German Credit...")
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
    print(mcr_df)

if __name__ == "__main__":
    generate_german_credit()

