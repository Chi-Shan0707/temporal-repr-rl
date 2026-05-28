"""
environments/wrappers.py — Observation wrappers for temporal encoding experiments.

Takes raw observations from FrozenLake/DoorGrid/SokobanGate and produces
flat observation vectors with configurable time encoding appended.

Encoding modes:
  'none'       → obs unchanged (agent must learn time from recurrence)
  'raw'        → append [t / max_t]
  'sinusoidal' → append [sin(2πt·f₁), cos(2πt·f₁), ...] (multi-frequency)
  'phase'      → append [sin(2π·phase/T), cos(2π·phase/T)] (circular)
  'onehot'     → append one-hot(phase, T)
"""

from __future__ import annotations

import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# Import existing temporal environments
sys.path.insert(0, "/mnt/d/CS/ReinforcementLearning/undo_gap")
from reproduction.experiments.temporal_environments import (
    FrozenLake, DoorGridLarge, SokobanGate,
    UP, DOWN, LEFT, RIGHT,
)


def sinusoidal_encode(t: int, dim: int, base: float = 10000.0) -> np.ndarray:
    """Transformer-style positional encoding: sin/cos with geometrically decaying frequencies."""
    pe = np.zeros(dim, dtype=np.float32)
    for i in range(dim // 2):
        freq = base ** (-2 * i / dim)
        pe[2 * i] = np.sin(t * freq)
        pe[2 * i + 1] = np.cos(t * freq)
    return pe


class TemporalObsWrapper:
    """Wraps a temporal environment to produce flat obs with time encoding.

    Not a gym.Wrapper because the underlying envs (FrozenLake etc.)
    are plain Python classes, not gymnasium.Envs.

    The underlying env must have:
      - .timestep property (int)
      - .PERIOD class attribute (int)
      - .mode attribute ('static' or 'temporal')
      - ._state_key() method (for getting the raw state)
      - 4 discrete actions
      - .reset() → state
      - .step(action) → (state, reward, terminated, truncated)
    """

    def __init__(
        self,
        env,
        time_encoding: str = "none",
        max_steps: int = 100,
        sinusoidal_dim: int = 8,
    ):
        self.env = env  # not using gym.Wrapper, store explicitly
        self.time_encoding = time_encoding
        self.max_steps = max_steps
        self.sinusoidal_dim = sinusoidal_dim
        self.period = getattr(env, "PERIOD", 1)
        self.n_actions = 4  # all environments have 4 discrete actions

        # Compute base obs dim from the environment
        base_dim = self._base_obs_dim()
        time_dim = self._time_dim()
        self.obs_dim = base_dim + time_dim

        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(base_dim + time_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(self.n_actions)

    def _base_obs_dim(self) -> int:
        """Infer base observation dimension from environment type."""
        if isinstance(self.env, FrozenLake):
            return self.env.ROWS * self.env.COLS  # 16
        elif isinstance(self.env, DoorGridLarge):
            return self.env.ROWS * self.env.COLS  # 64
        elif isinstance(self.env, SokobanGate):
            return self.env.ROWS * self.env.COLS + 2  # 38
        return 16  # fallback

    def _time_dim(self) -> int:
        """Number of additional features for the chosen encoding."""
        if self.time_encoding == "none":
            return 0
        elif self.time_encoding == "raw":
            return 1
        elif self.time_encoding == "sinusoidal":
            return self.sinusoidal_dim
        elif self.time_encoding == "phase":
            return 2
        elif self.time_encoding == "onehot":
            return self.period
        raise ValueError(f"Unknown encoding: {self.time_encoding}")

    def _get_base_obs(self) -> np.ndarray:
        """Get base observation as flat vector."""
        if isinstance(self.env, FrozenLake):
            obs = np.zeros(self.env.ROWS * self.env.COLS, dtype=np.float32)
            pos = self.env.agent_pos
            idx = pos[0] * self.env.COLS + pos[1]
            obs[idx] = 1.0
            return obs
        elif isinstance(self.env, DoorGridLarge):
            obs = np.zeros(self.env.ROWS * self.env.COLS, dtype=np.float32)
            pos = self.env.agent_pos
            idx = pos[0] * self.env.COLS + pos[1]
            obs[idx] = 1.0
            return obs
        elif isinstance(self.env, SokobanGate):
            # Encode as: grid_flat (walls/boxes/player/targets) + player_pos
            grid = np.zeros(self.env.ROWS * self.env.COLS, dtype=np.float32)
            for r in range(self.env.ROWS):
                for c in range(self.env.COLS):
                    if (r, c) in self.env.WALLS:
                        grid[r * self.env.COLS + c] = 0.2
            for box in self.env.boxes:
                grid[box[0] * self.env.COLS + box[1]] = 0.6
            grid[self.env.TARGET[0] * self.env.COLS + self.env.TARGET[1]] = 0.8
            pr, pc = self.env.player_pos
            pos = np.array(
                [pr / (self.env.ROWS - 1), pc / (self.env.COLS - 1)], dtype=np.float32
            )
            return np.concatenate([grid, pos])
        return np.zeros(16, dtype=np.float32)

    def _get_time_features(self) -> np.ndarray:
        """Get temporal encoding features."""
        t = self.env.timestep
        phase = t % self.period

        if self.time_encoding == "none":
            return np.array([], dtype=np.float32)
        elif self.time_encoding == "raw":
            return np.array([t / self.max_steps], dtype=np.float32)
        elif self.time_encoding == "sinusoidal":
            return sinusoidal_encode(t, self.sinusoidal_dim)
        elif self.time_encoding == "phase":
            return np.array([
                np.sin(2 * np.pi * phase / self.period),
                np.cos(2 * np.pi * phase / self.period),
            ], dtype=np.float32)
        elif self.time_encoding == "onehot":
            oh = np.zeros(self.period, dtype=np.float32)
            oh[phase] = 1.0
            return oh
        return np.array([], dtype=np.float32)

    def reset(self, **kwargs):
        state = self.env.reset()
        base = self._get_base_obs()
        time_feat = self._get_time_features()
        return np.concatenate([base, time_feat]), {}

    def step(self, action):
        state, reward, terminated, truncated = self.env.step(action)
        base = self._get_base_obs()
        time_feat = self._get_time_features()
        obs = np.concatenate([base, time_feat])

        # Reward shaping: small living penalty + goal bonus
        if isinstance(self.env, FrozenLake):
            if terminated and self.env.agent_pos == self.env.goal:
                reward = 1.0
            elif terminated:
                reward = -1.0
            else:
                reward = -0.01
        elif isinstance(self.env, DoorGridLarge):
            if terminated and self.env.agent_pos == self.env.GOAL:
                reward = 1.0
            else:
                reward = -0.01
        elif isinstance(self.env, SokobanGate):
            if terminated and self.env.TARGET in self.env.boxes:
                reward = 1.0
            else:
                reward = -0.01

        return obs, reward, terminated, truncated, {}

    @property
    def timestep(self) -> int:
        return self.env.timestep

    @property
    def phase(self) -> int:
        return self.env.timestep % self.period

    @property
    def physical_state(self):
        """Return the physical state (without time) for analysis."""
        return self.env._state_key()
