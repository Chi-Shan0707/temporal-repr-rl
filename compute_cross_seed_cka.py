from pathlib import Path
"""
Compute cross-seed CKA matrix for DoorGrid T=4.
Collects hidden states from each of 5 seeds and computes pairwise CKA.
"""

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
from environments.env_factory import make_env
from agents.lstm_policy import LSTMActorCritic
from temporal_repr.analysis.cka import linear_cka
from temporal_repr.environments.wrappers import TemporalObsWrapper
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from reproduction.experiments.temporal_environments import DoorGridLarge

SEEDS = [42, 123, 456, 789, 1024]
MODEL_DIR = "results/models/05_lstm_dg_temp_none_seed{}"
N_EPISODES = 200
MAX_STEPS = 200


def collect_hidden(model, env, n_episodes=200, max_steps=200):
    model.eval()
    device = next(model.parameters()).device
    all_hidden = []
    all_timesteps = []
    all_phases = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        hidden = model.init_hidden(batch_size=1, device=device)
        steps = 0
        while steps < max_steps:
            obs_t = torch.FloatTensor(obs).unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                logits, _, hidden_new, lstm_h = model(obs_t, hidden)
                h = lstm_h.squeeze(0).squeeze(0).cpu().numpy()
                hidden = hidden_new
            all_hidden.append(h)
            all_timesteps.append(env.timestep)
            all_phases.append(env.phase)
            action = logits.squeeze().argmax().item()
            obs, reward, terminated, truncated, _ = env.step(action)
            steps += 1
            if terminated or truncated:
                break

    model.train()
    return np.array(all_hidden), np.array(all_timesteps), np.array(all_phases)


def print_matrix(matrix, label=""):
    header = "        " + "  ".join(f"s{s:>4}" for s in SEEDS)
    print(header)
    for i, s in enumerate(SEEDS):
        row = "  ".join(f"{matrix[i,j]:.3f}" for j in range(len(SEEDS)))
        print(f"s{s:>4}  {row}")


def main():
    print("Collecting hidden states from 5 seeds (T=4)...")
    hidden_states = {}
    for seed in SEEDS:
        base_env = DoorGridLarge(mode="temporal", seed=seed)
        env = TemporalObsWrapper(base_env, time_encoding="none", max_steps=80)
        model = LSTMActorCritic(obs_dim=64, n_actions=4, hidden_dim=64)
        path = MODEL_DIR.format(seed) + "/best.pt"
        state = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model_state_dict"] if "model_state_dict" in state else state)
        h, t, p = collect_hidden(model, env, N_EPISODES, MAX_STEPS)
        hidden_states[seed] = {"hidden": h, "timesteps": t, "phases": p}
        print(f"  Seed {seed}: {h.shape[0]} steps collected")

    print("\nComputing 5x5 cross-seed CKA matrix (timestep-aligned means)...")
    n_seeds = len(SEEDS)
    cka_matrix_means = np.zeros((n_seeds, n_seeds))

    T = 4
    timestep_means = {}
    for seed in SEEDS:
        h = hidden_states[seed]["hidden"]
        t = hidden_states[seed]["timesteps"]
        means = np.zeros((T, h.shape[1]), dtype=np.float32)
        for ts in range(T):
            mask = t == ts
            if mask.any():
                means[ts] = h[mask].mean(axis=0)
        timestep_means[seed] = means

    for i, s1 in enumerate(SEEDS):
        for j, s2 in enumerate(SEEDS):
            m1 = timestep_means[s1]
            m2 = timestep_means[s2]
            cka_matrix_means[i, j] = linear_cka(m1, m2)

    print("Mean-hidden CKA (4 points, 64 dims):")
    print_matrix(cka_matrix_means)

    # Also compute CKA on full hidden states aligned by timestep
    print("\nComputing CKA on full hidden states (timestep-aligned, subsampled)...")
    cka_matrix_full = np.zeros((n_seeds, n_seeds))
    for i, s1 in enumerate(SEEDS):
        for j, s2 in enumerate(SEEDS):
            h1 = hidden_states[s1]["hidden"]
            h2 = hidden_states[s2]["hidden"]
            t1 = hidden_states[s1]["timesteps"]
            t2 = hidden_states[s2]["timesteps"]
            rng = np.random.RandomState(42)
            aligned_h1 = []
            aligned_h2 = []
            for ts in range(T):
                idx1 = np.where(t1 == ts)[0]
                idx2 = np.where(t2 == ts)[0]
                n_per = min(len(idx1), len(idx2), 80)
                sel1 = idx1[rng.choice(len(idx1), n_per, replace=False)]
                sel2 = idx2[rng.choice(len(idx2), n_per, replace=False)]
                aligned_h1.append(h1[sel1])
                aligned_h2.append(h2[sel2])
            aligned_h1 = np.concatenate(aligned_h1, axis=0)
            aligned_h2 = np.concatenate(aligned_h2, axis=0)
            cka_matrix_full[i, j] = linear_cka(aligned_h1, aligned_h2)

    print("Full-hidden CKA (320 points, 64 dims, timestep-aligned):")
    print_matrix(cka_matrix_full)

    # Use the means-based matrix as the primary result
    cka_mat = cka_matrix_means
    off_diag = [cka_mat[i,j] for i in range(n_seeds) for j in range(n_seeds) if i != j]
    print(f"\nMean-hidden off-diagonal: mean={np.mean(off_diag):.3f}, std={np.std(off_diag):.3f}, min={np.min(off_diag):.3f}, max={np.max(off_diag):.3f}")

    off_diag_full = [cka_matrix_full[i,j] for i in range(n_seeds) for j in range(n_seeds) if i != j]
    print(f"Full-hidden off-diagonal: mean={np.mean(off_diag_full):.3f}, std={np.std(off_diag_full):.3f}, min={np.min(off_diag_full):.3f}, max={np.max(off_diag_full):.3f}")

    results = {
        "seeds": SEEDS,
        "cka_matrix_means": cka_matrix_means.tolist(),
        "cka_matrix_full": cka_matrix_full.tolist(),
        "off_diag_means": {"mean": float(np.mean(off_diag)), "std": float(np.std(off_diag)), "min": float(np.min(off_diag)), "max": float(np.max(off_diag))},
        "off_diag_full": {"mean": float(np.mean(off_diag_full)), "std": float(np.std(off_diag_full)), "min": float(np.min(off_diag_full)), "max": float(np.max(off_diag_full))},
    }
    import json
    with open("figures/cross_seed_cka_matrix.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to figures/cross_seed_cka_matrix.json")


if __name__ == "__main__":
    main()
