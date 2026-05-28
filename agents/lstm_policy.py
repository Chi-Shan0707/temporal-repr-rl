"""
agents/lstm_policy.py — LSTM Actor-Critic for temporal RL experiments.

Key feature: exposes hidden states at every timestep for analysis.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


class LSTMActorCritic(nn.Module):
    """LSTM-based actor-critic for discrete action spaces.

    Architecture:
        obs → LSTM → actor_head → action logits
                      critic_head → state value

    The LSTM hidden state carries temporal information across timesteps.
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int = 4,
        hidden_dim: int = 64,
        n_layers: int = 1,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.lstm = nn.LSTM(
            input_size=obs_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=False,
        )
        self.actor = nn.Linear(hidden_dim, n_actions)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, obs_seq, hidden=None):
        """Forward pass on a sequence of observations.

        Args:
            obs_seq: (seq_len, batch, obs_dim)
            hidden: ((n_layers, batch, hidden_dim), (n_layers, batch, hidden_dim))
                    or None for zero init

        Returns:
            logits: (seq_len, batch, n_actions)
            values: (seq_len, batch, 1)
            new_hidden: (h, c) tuple
            lstm_out: (seq_len, batch, hidden_dim) — for analysis
        """
        if hidden is None:
            batch_size = obs_seq.size(1)
            device = obs_seq.device
            h0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim, device=device)
            c0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim, device=device)
            hidden = (h0, c0)

        lstm_out, new_hidden = self.lstm(obs_seq, hidden)
        logits = self.actor(lstm_out)
        values = self.critic(lstm_out)

        return logits, values, new_hidden, lstm_out

    def act(self, obs, hidden):
        """Select action for a single step. Returns (action, log_prob, value, new_hidden, lstm_h).

        obs: (1, 1, obs_dim)
        hidden: (h, c)
        """
        logits, values, new_hidden, lstm_h = self.forward(obs, hidden)
        dist = Categorical(logits=logits.squeeze(0).squeeze(0))
        action = dist.sample()
        log_prob = dist.log_prob(action)
        value = values.squeeze(0).squeeze(0)
        return action.item(), log_prob, value, new_hidden, lstm_h.squeeze(0).squeeze(0)

    def evaluate(self, obs_seq, actions, hidden=None):
        """Evaluate a full sequence. Returns (log_probs, values, entropy, lstm_out).

        obs_seq: (seq_len, batch, obs_dim)
        actions: (seq_len, batch) — long tensor
        """
        logits, values, _, lstm_out = self.forward(obs_seq, hidden)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, values.squeeze(-1), entropy, lstm_out

    def init_hidden(self, batch_size=1, device=None):
        """Create zero hidden state."""
        h = torch.zeros(self.n_layers, batch_size, self.hidden_dim, device=device)
        c = torch.zeros(self.n_layers, batch_size, self.hidden_dim, device=device)
        return (h, c)
