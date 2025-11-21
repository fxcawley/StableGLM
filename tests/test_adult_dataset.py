"""Integration test using the real-world Adult (Census Income) dataset."""

import csv
import os
import warnings
from typing import Any

import numpy as np
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from rashomon import RashomonSet

# Suppress warnings for cleaner test output
warnings.filterwarnings("ignore", category=UserWarning)


def load_adult_data(filepath: str, n_samples: int = 2000) -> tuple[np.ndarray, np.ndarray]:
    """Load and preprocess the Adult dataset without pandas."""
    
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            # Remove whitespace
            row = [x.strip() for x in row]
            # Skip rows with missing values '?'
            if '?' in row:
                continue
            data.append(row)
            if len(data) >= n_samples:
                break
    
    if not data:
        raise ValueError("No data loaded from file")

    # Convert to numpy array (object type initially)
    arr = np.array(data, dtype=object)
    
    # Columns (0-based):
    # Numerical: 0 (age), 2 (fnlwgt), 4 (education-num), 10 (cap-gain), 11 (cap-loss), 12 (hours)
    # Categorical: 1, 3, 5, 6, 7, 8, 9, 13
    # Target: 14
    
    # Extract features and target
    X_raw = arr[:, :-1]
    y_raw = arr[:, -1]
    
    # Encode target: >50K is 1, <=50K is 0
    y = np.array([1.0 if label == '>50K' else 0.0 for label in y_raw])
    
    # Feature indices
    numeric_features = [0, 2, 4, 10, 11, 12]
    categorical_features = [1, 3, 5, 6, 7, 8, 9, 13]
    
    # Preprocessing pipeline
    # Numeric: Impute mean (just in case), Scale
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # Categorical: OneHot
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    X = preprocessor.fit_transform(X_raw)
    
    return X, y


@pytest.mark.filterwarnings("ignore:Near-separation")
def test_adult_dataset_analysis() -> None:
    """End-to-end test on the Adult dataset."""
    
    # Path to the downloaded data
    data_path = os.path.join(os.path.dirname(__file__), "data", "adult.data")
    
    if not os.path.exists(data_path):
        pytest.skip(f"Adult dataset not found at {data_path}. Run scripts/download_data.py first.")
    
    # 1. Load Data
    try:
        X, y = load_adult_data(data_path, n_samples=1500)
    except Exception as e:
        pytest.fail(f"Failed to load data: {e}")
    
    print(f"\nLoaded Adult dataset: n={X.shape[0]}, d={X.shape[1]}")
    print(f"Prevalence (>50K): {np.mean(y):.2%}")
    
    # 2. Fit RashomonSet
    # Use stronger regularization because OneHot creates many features (d ~ 100)
    rs = RashomonSet(
        estimator="logistic",
        C=0.01,
        epsilon_mode="percent_loss",
        epsilon=0.05,
        random_state=42
    )
    rs.fit(X, y)
    
    # 3. Basic Assertions
    assert rs._fitted, "Model should be fitted"
    assert rs._theta_hat is not None, "Optimal weights should be computed"
    
    score = rs.score(X, y)
    print(f"Train accuracy: {score:.2%}")
    assert score > 0.75, "Model should have reasonable accuracy (>75%)"
    
    # 4. Diagnostics
    diag = rs.diagnostics()
    assert diag["L_hat"] > 0, "Loss should be positive"
    assert diag["epsilon"] > 0, "Epsilon should be positive"
    
    # 5. Predictive Multiplicity
    # Check ambiguity on a subset
    n_test = 100
    X_subset = X[:n_test]
    y_subset = y[:n_test]
    
    # Check Ambiguity
    mult = rs.multiplicity(
        X_subset,
        which=["ambiguity", "discrepancy"],
        y=y_subset,
        n_samples=50,
        threshold_mode="match_prevalence"
    )
    
    ambiguity = mult["ambiguity"]["ambiguity_rate"]
    discrepancy = mult["discrepancy"]["discrepancy_empirical"]
    
    print(f"Ambiguity rate: {ambiguity:.2%}")
    print(f"Discrepancy: {discrepancy:.2%}")
    
    # Assert that we get valid probabilities (0 to 1)
    assert 0.0 <= ambiguity <= 1.0
    assert 0.0 <= discrepancy <= 1.0
    
    # 6. VIC and Feature Importance
    # Use Ellipsoid for speed in test
    vic = rs.variable_importance_cloud(
        n_samples=50,
        sampler="ellipsoid",
        random_state=42
    )
    
    assert "mean" in vic
    assert "std" in vic
    assert len(vic["mean"]) == X.shape[1]
    
    # Check that at least some features have non-zero variance in VIC
    # (meaning the Rashomon set is not a single point)
    assert np.max(vic["std"]) > 1e-6, "Rashomon set should have volume (non-zero feature variance)"


if __name__ == "__main__":
    # Allow running this file directly
    try:
        test_adult_dataset_analysis()
        print("Test passed!")
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()

