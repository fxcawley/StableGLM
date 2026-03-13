"""Auto-generate and verify all numerical claims in tutorial.md.

Run from project root:
    python scripts/verify_tutorial_claims.py

Exit code 0 = all claims verified. Exit code 1 = mismatch found.
This script is run in CI to ensure tutorial numbers stay accurate.
"""
from __future__ import annotations

import re
import sys
import os

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rashomon import RashomonSet

TUTORIAL_PATH = os.path.join(os.path.dirname(__file__), "..", "docs", "examples", "tutorial.md")


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

    amb = rs.ambiguity(X_test, threshold_mode="fixed", threshold_value=0.5)
    disc = rs.discrepancy(X_test, n_samples=200, n_pairs=200, random_state=42)

    return {
        "n_test": len(X_test),
        "n_ambiguous": amb["n_ambiguous"],
        "ambiguity_rate": amb["ambiguity_rate"],
        "max_pair_disagreement": disc["max_pair_disagreement"],
    }


def parse_tutorial_claims() -> dict:
    """Extract claimed numbers directly from tutorial.md."""
    with open(TUTORIAL_PATH, encoding="utf-8") as f:
        text = f.read()

    claims = {}

    # Parse "Ambiguous: 39/171 (22.8%)" pattern
    m = re.search(r"Ambiguous:\s*(\d+)/(\d+)\s*\((\d+\.?\d*)%\)", text)
    if m:
        claims["n_ambiguous"] = int(m.group(1))
        claims["n_test"] = int(m.group(2))
        claims["ambiguity_rate"] = float(m.group(3)) / 100.0

    # Parse "Max pair disagreement: 7.6%" pattern
    m = re.search(r"Max pair disagreement:\s*(\d+\.?\d*)%", text)
    if m:
        claims["max_pair_disagreement"] = float(m.group(1)) / 100.0

    return claims


def verify_claims(results: dict, claims: dict) -> list[str]:
    """Check that generated results match the claims parsed from tutorial.md."""
    failures = []

    if "n_test" in claims and results["n_test"] != claims["n_test"]:
        failures.append(f"n_test: tutorial says {claims['n_test']}, got {results['n_test']}")

    if "n_ambiguous" in claims:
        diff = abs(results["n_ambiguous"] - claims["n_ambiguous"])
        if diff > 5:
            failures.append(
                f"n_ambiguous: tutorial says {claims['n_ambiguous']}, "
                f"got {results['n_ambiguous']} (diff {diff} > 5)"
            )

    if "ambiguity_rate" in claims:
        diff = abs(results["ambiguity_rate"] - claims["ambiguity_rate"])
        if diff > 0.05:
            failures.append(
                f"ambiguity_rate: tutorial says {claims['ambiguity_rate']:.1%}, "
                f"got {results['ambiguity_rate']:.1%} (diff {diff:.1%} > 5%)"
            )

    if "max_pair_disagreement" in claims:
        diff = abs(results["max_pair_disagreement"] - claims["max_pair_disagreement"])
        if diff > 0.03:
            failures.append(
                f"max_pair_disagreement: tutorial says {claims['max_pair_disagreement']:.1%}, "
                f"got {results['max_pair_disagreement']:.1%} (diff {diff:.1%} > 3%)"
            )

    return failures


if __name__ == "__main__":
    print("Generating tutorial results...")
    results = generate_tutorial_results()
    print(f"  n_test={results['n_test']}")
    print(f"  n_ambiguous={results['n_ambiguous']}")
    print(f"  ambiguity_rate={results['ambiguity_rate']:.1%}")
    print(f"  max_pair_disagreement={results['max_pair_disagreement']:.1%}")

    print("\nParsing claims from tutorial.md...")
    claims = parse_tutorial_claims()
    if not claims:
        print("  WARNING: Could not parse any claims from tutorial.md")
        sys.exit(1)
    for k, v in claims.items():
        print(f"  {k}={v}")

    print("\nVerifying...")
    failures = verify_claims(results, claims)

    if failures:
        print(f"\nFAILED: {len(failures)} claim(s) do not match:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("\nPASSED: All tutorial claims verified.")
        sys.exit(0)
