"""
analysis/report.py — Generate comprehensive analysis report + figures.

Runs all 5 analysis methods on collected hidden states and produces
a structured report with publication-quality figures.
"""

from __future__ import annotations

import sys
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from temporal_repr.analysis.linear_probe import linear_probe_timestep, linear_probe_phase
from temporal_repr.analysis.fourier import fourier_analysis
from temporal_repr.analysis.pca_vis import pca_analysis
from temporal_repr.analysis.cka import compare_encodings
from temporal_repr.analysis.mutual_info import estimate_mi_binning, estimate_mi_ksg


def full_analysis(
    data: dict,
    experiment_name: str = "",
    save_dir: str | None = None,
) -> dict:
    """Run all analyses on collected hidden states.

    Args:
        data: output from collect_hidden_states()
        experiment_name: label for figures
        save_dir: directory to save figures

    Returns:
        results dict with all analysis outputs
    """
    results = {"experiment": experiment_name}

    H = data["hidden_states"]
    T = data["timesteps"]
    P = data["phases"]
    period = data["period"]

    save_dir = Path(save_dir) if save_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Analysis: {experiment_name}")
    print(f"  Samples: {len(T)}, Hidden dim: {H.shape[1]}, Period: {period}")
    print(f"{'='*60}")

    # 1. Linear probe — timestep
    print("\n[1/6] Linear probe (timestep)...")
    probe_t = linear_probe_timestep(H, T)
    results["probe_timestep"] = {k: v for k, v in probe_t.items() if k != "coefficients"}
    print(f"  R² = {probe_t['r2']:.3f} (shuffled: {probe_t['r2_shuffled']:.3f})")

    # 2. Linear probe — phase
    print("[2/6] Linear probe (phase)...")
    probe_p = linear_probe_phase(H, P)
    results["probe_phase"] = probe_p
    print(f"  Accuracy = {probe_p['accuracy']:.3f} (shuffled: {probe_p['accuracy_shuffled']:.3f})")

    # 3. Fourier analysis
    print("[3/6] Fourier analysis...")
    fourier = fourier_analysis(H, T, period=period)
    if "error" in fourier:
        print(f"  Fourier analysis failed: {fourier['error']}")
        results["fourier"] = fourier
    else:
        results["fourier"] = {k: v for k, v in fourier.items() if k not in ("freqs", "avg_power")}
        print(f"  Dominant freqs: {fourier.get('dominant_freqs', 'N/A')}")
        env_pwr = fourier.get("env_period_power")
        print(f"  Env period power (f=1/{period}): {env_pwr:.4f}" if env_pwr else "  Env period power: N/A")
        print(f"  Spectral centroid: {fourier.get('spectral_centroid', 0):.4f}")

    # 4. PCA
    print("[4/6] PCA...")
    pca_path = str(save_dir / f"{experiment_name}_pca.png") if save_dir else None
    pca_res = pca_analysis(H, T, P, save_path=pca_path, title=experiment_name)
    results["pca"] = pca_res
    print(f"  PC1-timestep r = {pca_res['pc1_timestep_corr']:.3f}")
    print(f"  PC1 explains {pca_res['pc1_explained_variance']:.1%} variance")

    # 5. CKA
    print("[5/6] CKA comparison...")
    max_t = int(T.max()) + 1
    cka_res = compare_encodings(H, T, P, period, max_t=max_t)
    results["cka"] = cka_res
    best_enc = max(cka_res, key=cka_res.get)
    print(f"  Best match: {best_enc} (CKA={cka_res[best_enc]:.3f})")
    for name, score in sorted(cka_res.items(), key=lambda x: -x[1]):
        print(f"    {name:20s}: {score:.3f}")

    # 6. Mutual information
    print("[6/6] Mutual information...")
    mi_bin = estimate_mi_binning(H, T)
    results["mi_binning"] = mi_bin
    print(f"  MI (binning) = {mi_bin['mi']:.3f} nats (normalized: {mi_bin['mi_normalized']:.3f})")

    try:
        mi_ksg = estimate_mi_ksg(H, T, k=5, subsample=3000)
        results["mi_ksg"] = mi_ksg
        print(f"  MI (KSG) = {mi_ksg['mi']:.3f} nats")
    except Exception as e:
        print(f"  KSG failed: {e}")

    # Generate Fourier figure
    if save_dir and "freqs" in fourier and "avg_power" in fourier:
        _plot_fourier(fourier, experiment_name, period, save_dir)

    # Generate CKA figure
    if save_dir:
        _plot_cka(cka_res, experiment_name, save_dir)

    # Save results
    if save_dir:
        with open(save_dir / f"{experiment_name}_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)

    return results


def _plot_fourier(fourier: dict, name: str, period: int, save_dir: Path):
    """Plot Fourier power spectrum with environment period annotation."""
    fig, ax = plt.subplots(figsize=(10, 5))

    freqs = fourier["freqs"]
    power = fourier["avg_power"]

    ax.plot(freqs, power, linewidth=1.5, color="#2196F3")
    ax.fill_between(freqs, power, alpha=0.2, color="#2196F3")

    # Annotate environment period
    if period:
        env_freq = 1.0 / period
        ax.axvline(env_freq, color="red", linestyle="--", linewidth=1.5,
                   label=f"env period f=1/{period}={env_freq:.3f}")

    # Annotate dominant frequencies
    for i, freq in enumerate(fourier.get("dominant_freqs", [])[:3]):
        ax.axvline(freq, color="green", linestyle=":", alpha=0.6,
                   label=f"peak {i+1}: f={freq:.3f}" if i < 3 else None)

    ax.set_xlabel("Frequency (cycles/step)")
    ax.set_ylabel("Power")
    ax.set_title(f"Fourier Spectrum — {name}")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 0.5)

    plt.tight_layout()
    fig.savefig(save_dir / f"{name}_fourier.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_cka(cka_res: dict, name: str, save_dir: Path):
    """Plot CKA comparison as bar chart."""
    fig, ax = plt.subplots(figsize=(8, 4))

    names = list(cka_res.keys())
    scores = [cka_res[n] for n in names]

    bars = ax.barh(names, scores, color="#4CAF50", alpha=0.8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("CKA Similarity")
    ax.set_title(f"Learned vs Engineered Encodings — {name}")

    for bar, score in zip(bars, scores):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                f"{score:.3f}", va="center", fontsize=9)

    plt.tight_layout()
    fig.savefig(save_dir / f"{name}_cka.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
