"""
analysis/collect.py — Collect (hidden_state, timestep, phase) pairs from trained agents.

Runs evaluation episodes and extracts the LSTM/MLP hidden representation
at every timestep for downstream analysis.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from environments.env_factory import make_env
from agents.lstm_policy import LSTMActorCritic
from agents.mlp_policy import MLPActorCritic


def collect_hidden_states(
    model,
    env,
    agent_type: str = "lstm",
    n_episodes: int = 100,
    max_steps: int = 200,
    device=None,
    greedy: bool = True,
) -> dict:
    """Collect hidden states from evaluation episodes.

    Returns dict with:
        hidden_states: (n_total_steps, hidden_dim)
        timesteps: (n_total_steps,) absolute timestep in episode
        phases: (n_total_steps,) phase = timestep % period
        positions: list — physical state at each step
        rewards: (n_total_steps,)
        episode_lengths: (n_episodes,)
        episode_rewards: (n_episodes,)
    """
    model.eval()
    period = env.period
    hidden_dim = model.hidden_dim
    if device is None:
        device = next(model.parameters()).device

    all_hidden = []
    all_timesteps = []
    all_phases = []
    all_positions = []
    all_rewards = []
    episode_lengths = []
    episode_rewards = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        hidden = model.init_hidden(batch_size=1, device=device)
        ep_reward = 0
        steps = 0

        while steps < max_steps:
            obs_t = torch.FloatTensor(obs).unsqueeze(0).unsqueeze(0).to(device)

            with torch.no_grad():
                if agent_type == "lstm":
                    logits, _, hidden_new, lstm_h = model(obs_t, hidden)
                    h = lstm_h.squeeze(0).squeeze(0).cpu().numpy()
                    hidden = hidden_new
                else:
                    logits, _, _, features = model(obs_t)
                    h = features.squeeze(0).squeeze(0).cpu().numpy()

            if greedy:
                action = logits.squeeze().argmax().item()
            else:
                probs = torch.softmax(logits.squeeze(), dim=-1)
                action = torch.multinomial(probs, 1).item()

            all_hidden.append(h)
            all_timesteps.append(env.timestep)
            all_phases.append(env.phase)
            all_positions.append(env.physical_state)

            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            all_rewards.append(reward)
            ep_reward += reward
            steps += 1

            if done:
                break

        episode_lengths.append(steps)
        episode_rewards.append(ep_reward)

    model.train()

    return {
        "hidden_states": np.array(all_hidden, dtype=np.float32),
        "timesteps": np.array(all_timesteps, dtype=np.int32),
        "phases": np.array(all_phases, dtype=np.int32),
        "positions": all_positions,
        "rewards": np.array(all_rewards, dtype=np.float32),
        "episode_lengths": np.array(episode_lengths, dtype=np.int32),
        "episode_rewards": np.array(episode_rewards, dtype=np.float32),
        "period": period,
        "hidden_dim": hidden_dim,
        "agent_type": agent_type,
    }


def save_collected(data: dict, path: str):
    """Save collected data to .npz file."""
    save_dict = {
        "hidden_states": data["hidden_states"],
        "timesteps": data["timesteps"],
        "phases": data["phases"],
        "rewards": data["rewards"],
        "episode_lengths": data["episode_lengths"],
        "episode_rewards": data["episode_rewards"],
    }
    meta = {
        "period": data["period"],
        "hidden_dim": data["hidden_dim"],
        "agent_type": data["agent_type"],
    }
    np.savez(path, **save_dict, meta=np.array(meta))
    print(f"Saved {len(data['timesteps'])} steps to {path}")


def load_collected(path: str) -> dict:
    """Load collected data from .npz file."""
    loaded = np.load(path, allow_pickle=True)
    return {
        "hidden_states": loaded["hidden_states"],
        "timesteps": loaded["timesteps"],
        "phases": loaded["phases"],
        "rewards": loaded["rewards"],
        "episode_lengths": loaded["episode_lengths"],
        "episode_rewards": loaded["episode_rewards"],
        "period": int(loaded["meta"][0]),
        "hidden_dim": int(loaded["meta"][1]),
        "agent_type": str(loaded["meta"][2]),
    }
