"""
agents/train.py — REINFORCE with baseline training loop.

Supports both LSTMActorCritic and MLPActorCritic.
Uses episode-level REINFORCE: collect trajectory, re-evaluate, compute loss.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, "/mnt/d/CS/ReinforcementLearning/undo_gap/temporal_repr")
sys.path.insert(0, "/mnt/d/CS/ReinforcementLearning/undo_gap")
from temporal_repr.environments.env_factory import make_env
from temporal_repr.agents.lstm_policy import LSTMActorCritic
from temporal_repr.agents.mlp_policy import MLPActorCritic


def train(
    env_name: str,
    mode: str = "temporal",
    time_encoding: str = "none",
    agent_type: str = "lstm",
    n_episodes: int = 10000,
    hidden_dim: int = 64,
    lr: float = 1e-3,
    gamma: float = 0.99,
    entropy_coef: float = 0.01,
    eval_interval: int = 500,
    seed: int = 42,
    save_dir: str | None = None,
    verbose: bool = True,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = make_env(env_name, mode=mode, time_encoding=time_encoding, seed=seed)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    if agent_type == "lstm":
        model = LSTMActorCritic(obs_dim, n_actions, hidden_dim)
    elif agent_type == "mlp":
        model = MLPActorCritic(obs_dim, n_actions, hidden_dim)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    log = defaultdict(list)
    best_eval = -float("inf")

    for ep in range(n_episodes):
        model.train()
        obs, _ = env.reset()
        hidden = model.init_hidden(batch_size=1, device=device)

        # Collect trajectory
        trajectory_obs = []
        trajectory_actions = []
        trajectory_rewards = []
        done = False
        steps = 0

        while not done and steps < 200:
            trajectory_obs.append(obs.copy())
            obs_t = torch.FloatTensor(obs).unsqueeze(0).unsqueeze(0).to(device)

            with torch.no_grad():
                if agent_type == "lstm":
                    logits, _, hidden, _ = model(obs_t, hidden)
                else:
                    logits, _, _, _ = model(obs_t)

            dist = torch.distributions.Categorical(logits=logits.squeeze())
            action = dist.sample().item()

            trajectory_actions.append(action)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            trajectory_rewards.append(reward)
            steps += 1

        if steps == 0:
            continue

        # Re-evaluate with gradients (batch forward pass)
        obs_tensor = torch.FloatTensor(np.array(trajectory_obs)).unsqueeze(1).to(device)  # (T, 1, obs)
        act_tensor = torch.LongTensor(trajectory_actions).unsqueeze(1).to(device)          # (T, 1)

        if agent_type == "lstm":
            # Need to re-run LSTM step by step to get proper hidden states
            hidden = model.init_hidden(batch_size=1, device=device)
            log_probs_list = []
            values_list = []
            entropies_list = []
            for t in range(len(trajectory_obs)):
                obs_t = obs_tensor[t:t+1]  # (1, 1, obs)
                logits_t, val_t, hidden, _ = model(obs_t, hidden)
                dist_t = torch.distributions.Categorical(logits=logits_t.squeeze())
                log_probs_list.append(dist_t.log_prob(act_tensor[t].squeeze()))
                values_list.append(val_t.squeeze())
                entropies_list.append(dist_t.entropy())
        else:
            logits, values, _, _ = model(obs_tensor)
            dist = torch.distributions.Categorical(logits=logits)
            log_probs_list = [dist.log_prob(act_tensor[t].squeeze())[0] for t in range(steps)]
            values_list = [values[t].squeeze() for t in range(steps)]
            entropies_list = [dist.entropy()[t] for t in range(steps)]

        log_probs = torch.stack(log_probs_list)
        values = torch.stack(values_list)
        entropy = torch.stack(entropies_list).mean()

        # Compute returns
        returns = []
        G = 0
        for r in reversed(trajectory_rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns).to(device)

        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        advantages = returns - values.detach()
        policy_loss = -(log_probs * advantages).sum()
        value_loss = nn.functional.mse_loss(values, returns)
        loss = policy_loss + 0.5 * value_loss - entropy_coef * entropy

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        ep_reward = sum(trajectory_rewards)
        log["episode_rewards"].append(ep_reward)
        log["episode_lengths"].append(steps)

        if (ep + 1) % eval_interval == 0:
            eval_reward = evaluate(model, env, agent_type, n_episodes=20, device=device)
            log["eval_rewards"].append(eval_reward)
            if eval_reward > best_eval:
                best_eval = eval_reward
                if save_dir:
                    Path(save_dir).mkdir(parents=True, exist_ok=True)
                    torch.save(model.state_dict(), Path(save_dir) / "best.pt")

            if verbose:
                avg_r = np.mean(log["episode_rewards"][-eval_interval:])
                print(
                    f"Ep {ep+1:5d} | avg_reward={avg_r:+.3f} | "
                    f"eval={eval_reward:+.3f} | len={steps}"
                )

    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), Path(save_dir) / "final.pt")

    return model, dict(log)


def evaluate(model, env, agent_type, n_episodes=20, device=None):
    model.eval()
    total_rewards = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        hidden = model.init_hidden(batch_size=1, device=device)
        ep_reward = 0
        done = False
        steps = 0

        while not done and steps < 200:
            obs_t = torch.FloatTensor(obs).unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                if agent_type == "lstm":
                    logits, _, hidden, _ = model(obs_t, hidden)
                else:
                    logits, _, _, _ = model(obs_t)

            action = logits.squeeze().argmax().item()
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            steps += 1

        total_rewards.append(ep_reward)

    model.train()
    return np.mean(total_rewards)
