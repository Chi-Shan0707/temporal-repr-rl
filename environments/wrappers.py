"""
environments/wrappers.py — Observation wrappers for temporal encoding experiments.

Takes raw observations from FrozenLake/DoorGrid/SokobanGate/CyclicCorridor and
produces flat observation vectors with configurable time encoding appended.

Encoding modes:
  'none'       → obs unchanged (agent must learn time from recurrence)
  'raw'        → append [t / max_t]
  'sinusoidal' → append [sin(2πt·f₁), cos(2πt·f₁), ...] (multi-frequency)
  'phase'      → append [sin(2π·phase/T), cos(2π·phase/T)] (circular)
  'onehot'     → append one-hot(phase, T)
"""

from __future__ import annotations
from pathlib import Path

import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from reproduction.experiments.temporal_environments import (
    FrozenLake, DoorGridLarge, SokobanGate, CyclicCorridor,
)


def sinusoidal_encode(t: int, dim: int, base: float = 10000.0) -> np.ndarray:
    """Transformer-style positional encoding: sin/cos with geometrically decaying frequencies."""
    pe = np.zeros(dim, dtype=np.float32)
    for i in range(dim // 2):
        freq = base ** (-2 * i / dim)
        pe[2 * i] = np.sin(t * freq)
        pe[2 * i + 1] = np.cos(t * freq)
    return pe


def _is_2d_grid(env) -> bool:
    """Check if environment uses 2D grid positions (tuple) vs 1D (int)."""
    return hasattr(env, 'ROWS') and hasattr(env, 'COLS')


def _n_actions(env) -> int:
    """Infer number of actions from environment."""
    if hasattr(env, 'N_ACTIONS'):
        return env.N_ACTIONS
    return 4  # default: UP, DOWN, LEFT, RIGHT


def _obs_dim(env) -> int:
    """Infer base observation dimension from environment."""
    if isinstance(env, CyclicCorridor):
        return env.N_CELLS
    if isinstance(env, SokobanGate):
        return env.ROWS * env.COLS + 2
    if _is_2d_grid(env):
        return env.ROWS * env.COLS
    return 16  # fallback


def _onehot_pos(env) -> np.ndarray:
    """Get one-hot position encoding, works for both 1D and 2D environments."""
    dim = _obs_dim(env)
    obs = np.zeros(dim, dtype=np.float32)

    if isinstance(env, CyclicCorridor):
        obs[env.agent_pos] = 1.0
    elif isinstance(env, SokobanGate):
        # Grid + player position
        grid = np.zeros(env.ROWS * env.COLS, dtype=np.float32)
        for r in range(env.ROWS):
            for c in range(env.COLS):
                if (r, c) in env.WALLS:
                    grid[r * env.COLS + c] = 0.2
        for box in env.boxes:
            grid[box[0] * env.COLS + box[1]] = 0.6
        grid[env.TARGET[0] * env.COLS + env.TARGET[1]] = 0.8
        pr, pc = env.player_pos
        pos = np.array([pr / (env.ROWS - 1), pc / (env.COLS - 1)], dtype=np.float32)
        return np.concatenate([grid, pos])
    else:
        # 2D grid: FrozenLake, DoorGridLarge
        pos = env.agent_pos
        idx = pos[0] * env.COLS + pos[1]
        obs[idx] = 1.0
    return obs


def _reward_shaped(env, terminated: float) -> float:
    """Compute shaped reward. Works for all environments via duck typing."""
    goal_attr = None
    for attr in ('GOAL', 'goal', 'TARGET'):
        if hasattr(env, attr):
            goal_attr = attr
            break

    if goal_attr == 'TARGET':
        # SokobanGate: goal is a box position
        if terminated and env.TARGET in env.boxes:
            return 1.0
    elif goal_attr:
        goal = getattr(env, goal_attr)
        agent = env.agent_pos if hasattr(env, 'agent_pos') else env.player_pos
        if terminated and agent == goal:
            return 1.0

    return -0.01 if not terminated else -1.0


class TemporalObsWrapper:
    """Wraps a temporal environment to produce flat obs with time encoding.

    Works with any environment that exposes:
      - .timestep (int)
      - .PERIOD (int class attribute)
      - .mode ('static' or 'temporal')
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
        self.env = env
        self.time_encoding = time_encoding
        self.max_steps = max_steps
        self.sinusoidal_dim = sinusoidal_dim
        self.period = getattr(env, "PERIOD", 1)
        self.n_actions = _n_actions(env)

        base_dim = _obs_dim(env)
        time_dim = self._time_dim()
        self.obs_dim = base_dim + time_dim

        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(self.n_actions)

    def _time_dim(self) -> int:
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

    def _get_time_features(self) -> np.ndarray:
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
        self.env.reset()
        base = _onehot_pos(self.env)
        time_feat = self._get_time_features()
        return np.concatenate([base, time_feat]), {}

    def step(self, action):
        state, reward, terminated, truncated = self.env.step(action)
        base = _onehot_pos(self.env)
        time_feat = self._get_time_features()
        obs = np.concatenate([base, time_feat])
        reward = _reward_shaped(self.env, terminated)
        return obs, reward, terminated, truncated, {}

    @property
    def timestep(self) -> int:
        return self.env.timestep

    @property
    def phase(self) -> int:
        return self.env.timestep % self.period

    @property
    def physical_state(self):
        if hasattr(self.env, '_state_key'):
            return self.env._state_key()
        return self.env.agent_pos
