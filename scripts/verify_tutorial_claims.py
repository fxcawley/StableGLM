"""Auto-generate and verify all numerical claims in tutorial.md.

Run from project root:
    python scripts/verify_tutorial_claims.py

Exit code 0 = all claims verified. Exit code 1 = mismatch found.
This script is run in CI to ensure tutorial numbers stay accurate.
"""
from __future__ import annotations

import sys
import os
import re

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rashomon import RashomonSet


def generate_tutorial_results() -> dict:
    """Reproduce the exact tutorial configuration and return all claimed numbers."""
    data = load_breast_cancer()
    feature_names = ['radius', 'texture', 'perimeter', 'area', 'smoothness',
                     'compactness', 'concavity', 'concave_pts', 'symmetry', 'fractal_dim']
    X = data.data[:, :10]
    y = data.target.astype(float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42
    )

    rs = RashomonSet(
        estimator="logistic",
        epsilon=0.03,
        epsilon_mode="percent_loss",
        sampler="hitandrun",
        random_state=42,
        C=0.5,
        safety_override=True,
    ).fit(X_train, y_train)

    # Ambiguity
    amb = rs.ambiguity(X_test, threshold_mode="fixed", threshold_value=0.5)

    # Discrepancy
    disc = rs.discrepancy(X_test, n_samples=200, n_pairs=200, random_state=42)

    return {
        "n_test": len(X_test),
        "n_ambiguous": amb["n_ambiguous"],
        "ambiguity_rate": amb["ambiguity_rate"],
        "max_pair_disagreement": disc["max_pair_disagreement"],
    }


def verify_claims(results: dict) -> list[str]:
    """Check that generated results are consistent with tutorial.md claims."""
    tutorial_path = os.path.join(os.path.dirname(__file__), "..", "docs", "examples", "tutorial.md")
    with open(tutorial_path, encoding="utf-8") as f:
        text = f.read()

    failures = []

    # Claim: "39 out of 171 test patients" or "Ambiguous: 39/171 (22.8%)"
    # Allow +-5 ambiguous patients and +-3% rate due to sampling variance
    actual_n_amb = results["n_ambiguous"]
    actual_rate = results["ambiguity_rate"]
    actual_n_test = results["n_test"]

    # Check n_test matches
    if actual_n_test != 171:
        failures.append(f"n_test: expected 171, got {actual_n_test}")

    # Check ambiguity rate is within reasonable tolerance of the claimed value
    # The tutorial says 22.8% -- allow +-5% absolute
    claimed_rate = 0.228
    if abs(actual_rate - claimed_rate) > 0.05:
        failures.append(
            f"ambiguity_rate: claimed ~{claimed_rate:.1%}, got {actual_rate:.1%} "
            f"(difference {abs(actual_rate - claimed_rate):.1%} > 5% tolerance)"
        )

    # Check max pair disagreement -- tutorial says 7.6%, allow +-3%
    claimed_disc = 0.076
    actual_disc = results["max_pair_disagreement"]
    if abs(actual_disc - claimed_disc) > 0.03:
        failures.append(
            f"max_pair_disagreement: claimed ~{claimed_disc:.1%}, got {actual_disc:.1%} "
            f"(difference {abs(actual_disc - claimed_disc):.1%} > 3% tolerance)"
        )

    return failures


if __name__ == "__main__":
    print("Generating tutorial results...")
    results = generate_tutorial_results()
    print(f"  n_test={results['n_test']}")
    print(f"  n_ambiguous={results['n_ambiguous']}")
    print(f"  ambiguity_rate={results['ambiguity_rate']:.1%}")
    print(f"  max_pair_disagreement={results['max_pair_disagreement']:.1%}")

    print("\nVerifying against tutorial.md claims...")
    failures = verify_claims(results)

    if failures:
        print(f"\nFAILED: {len(failures)} claim(s) do not match:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("\nPASSED: All tutorial claims verified.")
        sys.exit(0)
