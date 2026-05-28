"""
experiments/compare_results.py — Cross-experiment comparison analysis.

Generates publication-quality comparison figures across all experiments.
"""

from __future__ import annotations

import sys
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

BASE_DIR = Path("/mnt/d/CS/ReinforcementLearning/undo_gap/temporal_repr")
FIGURES_DIR = BASE_DIR / "figures"


def load_all_results() -> dict:
    """Load all available experiment results."""
    results = {}
    for d in sorted(FIGURES_DIR.iterdir()):
        if d.is_dir() and not d.name.startswith("test"):
            json_path = d / f"{d.name}_results.json"
            if json_path.exists():
                with open(json_path) as f:
                    results[d.name] = json.load(f)
    return results


def summary_table(results: dict):
    """Print formatted comparison table."""
    print(f"\n{'='*110}")
    print(f"  CROSS-EXPERIMENT COMPARISON")
    print(f"{'='*110}")
    header = (f"  {'Experiment':<30} {'R²(t)':>6} {'Acc(φ)':>7} {'PC1-r':>6} "
              f"{'PC1-var':>8} {'Fourier':>10} {'Best CKA':>18} {'CKA score':>9}")
    print(header)
    print(f"  {'-'*30} {'-'*6} {'-'*7} {'-'*6} {'-'*8} {'-'*10} {'-'*18} {'-'*9}")

    for name, res in sorted(results.items()):
        r2 = res.get("probe_timestep", {}).get("r2", 0)
        r2_shuf = res.get("probe_timestep", {}).get("r2_shuffled", 0)
        acc = res.get("probe_phase", {}).get("accuracy", 0)
        pc1r = res.get("pca", {}).get("pc1_timestep_corr", 0)
        pc1v = res.get("pca", {}).get("pc1_explained_variance", 0)
        centroid = res.get("fourier", {}).get("spectral_centroid", None)
        fourier_str = f"{centroid:.4f}" if centroid else "FAILED"

        cka_scores = res.get("cka", {})
        if cka_scores:
            best_enc = max(cka_scores, key=cka_scores.get)
            best_score = cka_scores[best_enc]
        else:
            best_enc = "N/A"
            best_score = 0

        print(f"  {name:<30} {r2:>6.3f} {acc:>7.3f} {pc1r:>6.3f} "
              f"{pc1v:>7.1%} {fourier_str:>10} {best_enc:>18} {best_score:>9.3f}")

    print()


def plot_r2_comparison(results: dict, save_path: Path):
    """Bar chart comparing R² across experiments."""
    fig, ax = plt.subplots(figsize=(12, 5))

    names = sorted(results.keys())
    r2_vals = [results[n].get("probe_timestep", {}).get("r2", 0) for n in names]
    r2_shuf = [results[n].get("probe_timestep", {}).get("r2_shuffled", 0) for n in names]

    x = np.arange(len(names))
    width = 0.35

    bars1 = ax.bar(x - width/2, r2_vals, width, label="R² (timestep)", color="#2196F3", alpha=0.85)
    bars2 = ax.bar(x + width/2, r2_shuf, width, label="R² (shuffled)", color="#FF9800", alpha=0.85)

    ax.set_ylabel("R² Score")
    ax.set_title("Linear Probe: Can Hidden States Predict Timestep?")
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=7, rotation=0)
    ax.legend()
    ax.set_ylim(-0.1, 1.1)
    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5)

    for bar, val in zip(bars1, r2_vals):
        if val > 0.01:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_cka_heatmap(results: dict, save_path: Path):
    """Heatmap of CKA similarities across experiments."""
    encodings = ["raw_scalar", "sinusoidal_pe", "phase_sin_cos", "onehot_phase", "remaining_time"]
    names = sorted(results.keys())

    if not names:
        return

    matrix = np.zeros((len(names), len(encodings)))
    for i, name in enumerate(names):
        cka = results[name].get("cka", {})
        for j, enc in enumerate(encodings):
            matrix[i, j] = cka.get(enc, 0)

    fig, ax = plt.subplots(figsize=(10, max(4, len(names) * 0.8 + 2)))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(np.arange(len(encodings)))
    ax.set_yticks(np.arange(len(names)))
    ax.set_xticklabels([e.replace("_", " ") for e in encodings], rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels([n.replace("_", "\n") for n in names], fontsize=8)

    for i in range(len(names)):
        for j in range(len(encodings)):
            val = matrix[i, j]
            color = "white" if val > 0.7 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9, color=color)

    ax.set_title("CKA Similarity: Learned vs. Engineered Temporal Encodings")
    fig.colorbar(im, ax=ax, label="CKA Similarity")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_phase_accuracy(results: dict, save_path: Path):
    """Bar chart of phase classification accuracy."""
    fig, ax = plt.subplots(figsize=(10, 5))

    names = sorted(results.keys())
    acc = [results[n].get("probe_phase", {}).get("accuracy", 0) for n in names]
    acc_shuf = [results[n].get("probe_phase", {}).get("accuracy_shuffled", 0) for n in names]

    x = np.arange(len(names))
    width = 0.35

    colors = []
    for n in names:
        if "mlp" in n and "none" in n:
            colors.append("#FF5722")   # orange-red for MLP without time
        elif "mlp" in n:
            colors.append("#4CAF50")   # green for MLP with time
        elif "static" in n:
            colors.append("#9E9E9E")   # gray for static
        else:
            colors.append("#2196F3")   # blue for temporal LSTM

    bars1 = ax.bar(x - width/2, acc, width, label="Phase Accuracy", color=colors, alpha=0.85)
    bars2 = ax.bar(x + width/2, acc_shuf, width, label="Shuffled Baseline", color="#BDBDBD", alpha=0.85)

    ax.set_ylabel("Classification Accuracy")
    ax.set_title("Phase Prediction from Hidden States")
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=7)
    ax.legend()
    ax.set_ylim(-0.05, 1.1)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_pca_variance(results: dict, save_path: Path):
    """Compare PCA explained variance across experiments."""
    fig, ax = plt.subplots(figsize=(10, 5))

    names = sorted(results.keys())
    x = np.arange(len(names))

    pc1_var = []
    for n in names:
        evr = results[n].get("pca", {}).get("explained_variance_ratio", [])
        pc1_var.append(evr[0] if evr else 0)

    bars = ax.bar(x, pc1_var, color="#673AB7", alpha=0.85)
    ax.set_ylabel("PC1 Explained Variance Ratio")
    ax.set_title("How Much Variance Does the Temporal Axis Capture?")
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=7)
    ax.set_ylim(0, 1.05)

    for bar, val in zip(bars, pc1_var):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{val:.2f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_mi_comparison(results: dict, save_path: Path):
    """Compare mutual information across experiments."""
    fig, ax = plt.subplots(figsize=(10, 5))

    names = sorted(results.keys())
    x = np.arange(len(names))

    mi_vals = [results[n].get("mi_binning", {}).get("mi", 0) for n in names]
    mi_norm = [results[n].get("mi_binning", {}).get("mi_normalized", 0) for n in names]

    width = 0.35
    bars1 = ax.bar(x - width/2, mi_vals, width, label="MI (nats)", color="#00BCD4", alpha=0.85)
    bars2 = ax.bar(x + width/2, mi_norm, width, label="MI (normalized)", color="#E91E63", alpha=0.85)

    ax.set_ylabel("Mutual Information")
    ax.set_title("I(hidden_state; timestep) — How Much Temporal Info Is Encoded?")
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=7)
    ax.legend()
    ax.set_ylim(0, max(max(mi_vals, default=1), max(mi_norm, default=1)) * 1.2)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def generate_all_figures():
    """Generate all comparison figures."""
    print("\n" + "="*60)
    print("  CROSS-EXPERIMENT COMPARISON ANALYSIS")
    print("="*60)

    results = load_all_results()
    print(f"\n  Loaded {len(results)} experiments: {list(results.keys())}")

    if not results:
        print("  No results found!")
        return

    summary_table(results)

    comp_dir = FIGURES_DIR / "comparison"
    comp_dir.mkdir(parents=True, exist_ok=True)

    print("  Generating comparison figures...")
    plot_r2_comparison(results, comp_dir / "r2_comparison.png")
    plot_cka_heatmap(results, comp_dir / "cka_heatmap.png")
    plot_phase_accuracy(results, comp_dir / "phase_accuracy.png")
    plot_pca_variance(results, comp_dir / "pca_variance.png")
    plot_mi_comparison(results, comp_dir / "mi_comparison.png")

    print(f"\n  All comparison figures saved to {comp_dir}/")
    return results


if __name__ == "__main__":
    generate_all_figures()
