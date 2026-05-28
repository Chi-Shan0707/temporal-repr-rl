"""Generate 3 core figures for the restructured paper."""
import sys; sys.path.insert(0, '.')
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from sklearn.decomposition import PCA

FIGDIR = Path(__file__).resolve().parent / "figures"
FIGDIR.mkdir(exist_ok=True)
DATADIR = Path(__file__).resolve().parent.parent / "figures"

SEEDS = [42, 123, 456, 789, 1024]
ENCODING_NAMES = ["raw_scalar", "sinusoidal_pe", "phase_sin_cos", "onehot_phase"]
ENCODING_LABELS = ["Raw scalar", "Sinusoidal PE", "Phase sin/cos", "One-hot phase"]

def load_result(exp_name):
    p = DATADIR / exp_name / f"{exp_name}_results.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)

# ============================================================
# FIGURE 1: CKA Heatmap — 5 seeds × 4 encodings (DoorGrid)
# ============================================================
print("=== Figure 1: CKA Degeneracy Heatmap ===")

cka_matrix = []
for seed in SEEDS:
    exp = f"05_lstm_dg_temp_none_seed{seed}"
    r = load_result(exp)
    if r is None:
        cka_matrix.append([np.nan]*4)
        continue
    row = [r["cka"].get(en, 0) for en in ENCODING_NAMES]
    cka_matrix.append(row)
cka_matrix = np.array(cka_matrix)

fig, ax = plt.subplots(figsize=(7, 3.5))
cmap = plt.cm.YlOrRd
im = ax.imshow(cka_matrix, cmap=cmap, vmin=0, vmax=1, aspect="auto")

for i in range(len(SEEDS)):
    for j in range(len(ENCODING_LABELS)):
        val = cka_matrix[i, j]
        color = "white" if val > 0.6 else "black"
        ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=11, color=color, fontweight="bold")

ax.set_xticks(range(len(ENCODING_LABELS)))
ax.set_xticklabels(ENCODING_LABELS, fontsize=10)
ax.set_yticks(range(len(SEEDS)))
ax.set_yticklabels([f"Seed {s}" for s in SEEDS], fontsize=10)
ax.set_title("CKA: DoorGrid Hidden States vs. Engineered Encodings", fontsize=12, fontweight="bold")

cbar = plt.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label("CKA Similarity", fontsize=10)

best_idx = np.argmax(cka_matrix, axis=1)
for i, j in enumerate(best_idx):
    rect = plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=False, edgecolor="blue", linewidth=2.5)
    ax.add_patch(rect)

plt.tight_layout()
fig.savefig(FIGDIR / "fig1_cka_degeneracy_heatmap.pdf", dpi=300, bbox_inches="tight")
fig.savefig(FIGDIR / "fig1_cka_degeneracy_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig1_cka_degeneracy_heatmap")

# ============================================================
# FIGURE 2: Trained vs Untrained + Complexity Gradient
# ============================================================
print("\n=== Figure 2: Trained vs Untrained ===")

envs = ["FrozenLake", "DoorGrid", "SokobanGate"]
trained_files = ["01_lstm_fl_temp_none", "05_lstm_dg_temp_none", "06_lstm_sk_temp_none"]
untrained_files = ["untrained_lstm_frozenlake", "untrained_lstm_doorgrid", "untrained_lstm_sokoban_gate"]

metrics = {"R²(t)": [], "CKA Best": [], "Phase Acc": []}
untrained_metrics = {"R²(t)": [], "CKA Best": [], "Phase Acc": []}

for tf, uf in zip(trained_files, untrained_files):
    tr = load_result(tf)
    ur = load_result(uf)
    if tr:
        metrics["R²(t)"].append(tr["probe_timestep"]["r2"])
        metrics["CKA Best"].append(max(tr["cka"].values()))
        metrics["Phase Acc"].append(tr["probe_phase"]["accuracy"])
    if ur:
        untrained_metrics["R²(t)"].append(ur["probe_timestep"]["r2"])
        untrained_metrics["CKA Best"].append(max(ur["cka"].values()))
        untrained_metrics["Phase Acc"].append(ur["probe_phase"]["accuracy"])

fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
x = np.arange(len(envs))
width = 0.35

colors_trained = "#2166ac"
colors_untrained = "#b2182b"

for idx, (metric_name, ax) in enumerate(zip(["R²(t)", "CKA Best", "Phase Acc"], axes)):
    tr_vals = metrics[metric_name]
    ut_vals = untrained_metrics[metric_name]
    bars1 = ax.bar(x - width/2, tr_vals, width, label="Trained", color=colors_trained, edgecolor="white")
    bars2 = ax.bar(x + width/2, ut_vals, width, label="Untrained", color=colors_untrained, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(envs, fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel(metric_name, fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for bar_group in [bars1, bars2]:
        for bar in bar_group:
            h = bar.get_height()
            ax.annotate(f"{h:.2f}", xy=(bar.get_x() + bar.get_width()/2, h),
                       xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)

fig.suptitle("Temporal Encoding Is Learned, Not Architectural", fontsize=12, fontweight="bold", y=1.02)
plt.tight_layout()
fig.savefig(FIGDIR / "fig2_trained_vs_untrained.pdf", dpi=300, bbox_inches="tight")
fig.savefig(FIGDIR / "fig2_trained_vs_untrained.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig2_trained_vs_untrained")

# ============================================================
# FIGURE 3: T=4 vs T=8 Strategy Shift
# ============================================================
print("\n=== Figure 3: T=4 vs T=8 Strategy Shift ===")

r_t4 = load_result("05_lstm_dg_temp_none")
r_t8 = load_result("11_lstm_dg_t8_none")

enc_labels_short = ["Raw\nscalar", "Sinusoidal\nPE", "Phase\nsin/cos", "One-hot\nphase"]
cka_t4 = [r_t4["cka"][en] for en in ENCODING_NAMES] if r_t4 else [0]*4
cka_t8 = [r_t8["cka"][en] for en in ENCODING_NAMES] if r_t8 else [0]*4

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

x = np.arange(len(enc_labels_short))
width = 0.35

bars_t4 = ax1.bar(x - width/2, cka_t4, width, label="T = 4", color="#4393c3", edgecolor="white")
bars_t8 = ax1.bar(x + width/2, cka_t8, width, label="T = 8", color="#d6604d", edgecolor="white")

for bar_group in [bars_t4, bars_t8]:
    for bar in bar_group:
        h = bar.get_height()
        ax1.annotate(f"{h:.2f}", xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)

ax1.set_xticks(x)
ax1.set_xticklabels(enc_labels_short, fontsize=9)
ax1.set_ylabel("CKA Similarity", fontsize=10)
ax1.set_ylim(0, 1.0)
ax1.legend(fontsize=9)
ax1.set_title("Encoding Strategy Shifts with Period", fontsize=11, fontweight="bold")
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

pc1_t4 = r_t4["pca"]["pc1_timestep_corr"] if r_t4 else 0
pc1_t8 = r_t8["pca"]["pc1_timestep_corr"] if r_t8 else 0
pc1var_t4 = r_t4["pca"]["pc1_explained_variance"] * 100 if r_t4 else 0
pc1var_t8 = r_t8["pca"]["pc1_explained_variance"] * 100 if r_t8 else 0

categories = ["PC1-timestep\ncorrelation", "PC1 variance\n(%)"]
vals_t4 = [pc1_t4, pc1var_t4]
vals_t8 = [pc1_t8, pc1var_t8]

x2 = np.arange(len(categories))
bars_a = ax2.bar(x2 - width/2, vals_t4, width, label="T = 4", color="#4393c3", edgecolor="white")
bars_b = ax2.bar(x2 + width/2, vals_t8, width, label="T = 8", color="#d6604d", edgecolor="white")

for bar_group in [bars_a, bars_b]:
    for bar in bar_group:
        h = bar.get_height()
        ax2.annotate(f"{h:.1f}" if h > 1 else f"{h:.3f}", xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)

ax2.set_xticks(x2)
ax2.set_xticklabels(categories, fontsize=9)
ax2.set_ylim(0, 1.15)
ax2.legend(fontsize=9)
ax2.set_title("PC1 Becomes Monotonic at T = 8", fontsize=11, fontweight="bold")
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

plt.tight_layout()
fig.savefig(FIGDIR / "fig3_horizon_sensitivity.pdf", dpi=300, bbox_inches="tight")
fig.savefig(FIGDIR / "fig3_horizon_sensitivity.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig3_horizon_sensitivity")

print("\n=== All 3 figures generated ===")
