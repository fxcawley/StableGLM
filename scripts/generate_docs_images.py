import numpy as np
import matplotlib.pyplot as plt
import os
from rashomon import RashomonSet

def generate_images():
    output_dir = os.path.join("docs", "_static")
    os.makedirs(output_dir, exist_ok=True)

    # Reproducibility
    np.random.seed(42)

    # Generate synthetic data (n=200, d=5)
    n_samples = 200
    n_features = 5
    X = np.random.randn(n_samples, n_features)

    # Feature 0 and 1 are correlated
    X[:, 1] = X[:, 0] * 0.9 + np.random.randn(n_samples) * 0.1

    # True coefficients (Feature 2 is irrelevant)
    true_theta = np.array([2.0, -1.5, 0.0, 0.5, -0.5])

    # Generate labels
    logits = X @ true_theta
    probs = 1 / (1 + np.exp(-logits))
    y = (np.random.rand(n_samples) < probs).astype(float)

    # Initialize RashomonSet
    rs = RashomonSet(
        estimator="logistic",
        epsilon=0.05,
        epsilon_mode="percent_loss",
        sampler="hitandrun", 
        random_state=42
    )

    rs.fit(X, y)

    # 1. VIC Plot
    plt.figure(figsize=(8, 5))
    rs.plot_vic()
    plt.title("Variable Importance Cloud")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "tutorial_vic.png"))
    plt.close()
    print("Generated tutorial_vic.png")

    # 2. Ambiguity Plot
    plt.figure(figsize=(8, 5))
    rs.plot_ambiguity(X[:20], y=y[:20])
    plt.title("Predictive Multiplicity (First 20 Samples)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "tutorial_ambiguity.png"))
    plt.close()
    print("Generated tutorial_ambiguity.png")

if __name__ == "__main__":
    generate_images()

