"""
agents/mlp_policy.py — MLP Actor-Critic baseline (no temporal memory).

Same interface as LSTMActorCritic but uses FC layers instead of LSTM.
Hidden state is always None — no recurrence.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Categorical


class MLPActorCritic(nn.Module):
    """MLP actor-critic with no recurrent state.

    Architecture:
        obs → FC1 → ReLU → FC2 → ReLU → actor_head → logits
                                      → critic_head → value
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int = 4,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim

        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.actor = nn.Linear(hidden_dim, n_actions)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, obs_seq, hidden=None):
        """Forward pass. Ignores hidden state (no recurrence).

        Args:
            obs_seq: (seq_len, batch, obs_dim)
            hidden: ignored

        Returns:
            logits, values, None (no hidden), features (for analysis)
        """
        seq_len, batch, _ = obs_seq.shape
        flat = obs_seq.reshape(-1, self.obs_dim)
        features = self.shared(flat)
        logits = self.actor(features)
        values = self.critic(features)

        logits = logits.reshape(seq_len, batch, -1)
        values = values.reshape(seq_len, batch, 1)
        features = features.reshape(seq_len, batch, -1)

        return logits, values, None, features

    def act(self, obs, hidden=None):
        """Select action. obs: (1, 1, obs_dim)."""
        logits, values, _, features = self.forward(obs)
        dist = Categorical(logits=logits.squeeze(0).squeeze(0))
        action = dist.sample()
        log_prob = dist.log_prob(action)
        value = values.squeeze(0).squeeze(0)
        return action.item(), log_prob, value, None, features.squeeze(0).squeeze(0)

    def evaluate(self, obs_seq, actions, hidden=None):
        """Evaluate sequence."""
        logits, values, _, features = self.forward(obs_seq)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, values.squeeze(-1), entropy, features

    def init_hidden(self, batch_size=1, device=None):
        return None
