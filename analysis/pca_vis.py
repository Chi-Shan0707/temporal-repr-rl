"""
analysis/pca_vis.py — PCA visualization of hidden states colored by timestep/phase.

Answers: Is timestep the dominant axis of variation in the learned representation?
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def pca_analysis(
    hidden_states: np.ndarray,
    timesteps: np.ndarray,
    phases: np.ndarray,
    save_path: str | None = None,
    title: str = "",
) -> dict:
    """PCA analysis and visualization.

    Returns:
        pc1_timestep_corr: Pearson r between PC1 and timestep
        pc2_timestep_corr: Pearson r between PC2 and timestep
        explained_variance_ratio: first 10 PCs
        fig: matplotlib figure (if save_path is None)
    """
    pca = PCA(n_components=min(10, hidden_states.shape[1]))
    projected = pca.fit_transform(hidden_states)

    # Correlation with timestep
    pc1_corr = np.corrcoef(projected[:, 0], timesteps)[0, 1]
    pc2_corr = np.corrcoef(projected[:, 1], timesteps)[0, 1]

    # Phase correlation
    pc1_phase_corr = np.corrcoef(projected[:, 0], phases)[0, 1]

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: colored by timestep
    sc1 = axes[0].scatter(
        projected[:, 0], projected[:, 1],
        c=timesteps, cmap="viridis", s=2, alpha=0.5,
    )
    axes[0].set_xlabel(f"PC1 (r={pc1_corr:.2f} with t)")
    axes[0].set_ylabel(f"PC2 (r={pc2_corr:.2f} with t)")
    axes[0].set_title(f"{title} — colored by timestep")
    plt.colorbar(sc1, ax=axes[0], label="timestep")

    # Right: colored by phase
    n_phases = len(np.unique(phases))
    sc2 = axes[1].scatter(
        projected[:, 0], projected[:, 1],
        c=phases, cmap="tab10", s=2, alpha=0.5,
        vmin=0, vmax=max(n_phases - 1, 1),
    )
    axes[1].set_xlabel(f"PC1 (r={pc1_corr:.2f} with t)")
    axes[1].set_ylabel(f"PC2")
    axes[1].set_title(f"{title} — colored by phase")
    plt.colorbar(sc2, ax=axes[1], label="phase")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return {
        "pc1_timestep_corr": float(pc1_corr),
        "pc2_timestep_corr": float(pc2_corr),
        "pc1_phase_corr": float(pc1_phase_corr),
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "pc1_explained_variance": float(pca.explained_variance_ratio_[0]),
    }
