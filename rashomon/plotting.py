"""Visualization utilities for Rashomon sets (G32)."""

from typing import Any, Optional, Dict, Tuple, List
import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes
except ImportError:
    plt = None
    Figure = Any
    Axes = Any


def plot_vic(
    vic_result: Dict[str, Any],
    theta_hat: Optional[np.ndarray] = None,
    figsize: Optional[Tuple[float, float]] = None,
) -> Tuple[Figure, Axes]:
    """Plot Variable Importance Cloud (coefficient distributions).
    
    Parameters
    ----------
    vic_result : dict
        Result from RashomonSet.variable_importance_cloud().
    theta_hat : array, optional
        Optimal parameter vector to highlight.
    figsize : tuple, optional
        Figure size.
    
    Returns
    -------
    fig, ax : matplotlib objects
    """
    if plt is None:
        raise ImportError("matplotlib is required for plotting.")

    samples = vic_result["samples"]
    names = vic_result["feature_names"]
    intervals = vic_result["intervals"]
    mean = vic_result["mean"]

    d = samples.shape[1]

    # Create figure
    if figsize is None:
        figsize = (min(12, max(8, d * 1.2)), 6)
    fig, ax = plt.subplots(figsize=figsize)

    # Violin or box plots for each feature
    positions = np.arange(d)
    
    # Use violinplot for density estimation
    parts = ax.violinplot(
        [samples[:, j] for j in range(d)],
        positions=positions,
        widths=0.7,
        showmeans=False,
        showmedians=False,
    )

    # Color violins
    for pc in parts["bodies"]:
        pc.set_facecolor("#8dd3c7")
        pc.set_alpha(0.7)
        pc.set_edgecolor("black")
        pc.set_linewidth(1)

    # Add mean markers
    ax.scatter(positions, mean, color="red", s=60, zorder=3, label="Mean", marker="D")

    # Add theta_hat if requested
    if theta_hat is not None:
        ax.scatter(
            positions,
            theta_hat,
            color="blue",
            s=80,
            zorder=4,
            label="θ̂ (optimal)",
            marker="*",
        )

    # Add 90% intervals as error bars
    ax.errorbar(
        positions,
        mean,
        yerr=[mean - intervals[:, 0], intervals[:, 1] - mean],
        fmt="none",
        ecolor="gray",
        alpha=0.5,
        capsize=3,
        label="90% interval",
    )

    # Styling
    ax.set_xticks(positions)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("Coefficient Value")
    ax.set_xlabel("Feature")
    ax.set_title("Variable Importance Cloud (VIC)")
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    return fig, ax


def plot_ambiguity(
    margins: np.ndarray,
    threshold: float = 0.0,
    ambiguous_indices: Optional[np.ndarray] = None,
    bins: int = 30,
    figsize: Optional[Tuple[float, float]] = (10, 6),
) -> Tuple[Figure, Axes]:
    """Plot distribution of predictive margins and highlight ambiguity region.
    
    Parameters
    ----------
    margins : array
        Predictive margins (x^T theta_hat) for instances.
    threshold : float
        Decision threshold in margin space.
    ambiguous_indices : array, optional
        Indices of instances flagged as ambiguous.
    
    Returns
    -------
    fig, ax
    """
    if plt is None:
        raise ImportError("matplotlib is required for plotting.")

    fig, ax = plt.subplots(figsize=figsize)

    # Histogram of all margins
    counts, bin_edges, patches = ax.hist(
        margins, bins=bins, color="skyblue", edgecolor="black", alpha=0.7, label="All Instances"
    )
    
    # Highlight ambiguous instances if provided
    if ambiguous_indices is not None and len(ambiguous_indices) > 0:
        amb_margins = margins[ambiguous_indices]
        ax.hist(
            amb_margins, bins=bin_edges, color="orange", edgecolor="red", alpha=0.8, 
            label="Ambiguous Instances", hatch="//"
        )

    ax.axvline(threshold, color="red", linestyle="--", linewidth=2, label=f"Threshold τ={threshold:.2f}")
    
    ax.set_xlabel("Predictive Margin ($x^T \\hat{\\theta}$)")
    ax.set_ylabel("Count")
    ax.set_title("Predictive Ambiguity Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, ax


def plot_discrepancy(
    discrepancy_matrix: np.ndarray,
    sample_indices: Optional[List[int]] = None,
    figsize: Optional[Tuple[float, float]] = (8, 7),
) -> Tuple[Figure, Axes]:
    """Plot pairwise discrepancy matrix between sampled models.
    
    Parameters
    ----------
    discrepancy_matrix : array of shape (n_samples, n_samples)
        Pairwise disagreement rates.
    
    Returns
    -------
    fig, ax
    """
    if plt is None:
        raise ImportError("matplotlib is required for plotting.")

    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(discrepancy_matrix, cmap="Reds", vmin=0, vmax=np.max(discrepancy_matrix))
    plt.colorbar(im, ax=ax, label="Disagreement Rate")
    
    ax.set_title("Pairwise Model Discrepancy")
    ax.set_xlabel("Model Index $i$")
    ax.set_ylabel("Model Index $j$")
    
    plt.tight_layout()
    return fig, ax

