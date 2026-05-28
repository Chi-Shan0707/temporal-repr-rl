"""
environments/env_factory.py — Create environment instances with wrappers.

Usage:
    env = make_env('frozenlake', mode='temporal', time_encoding='phase')
    env = make_env('sokoban_gate', mode='static', time_encoding='none')
"""

from __future__ import annotations
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from .wrappers import TemporalObsWrapper
from reproduction.experiments.temporal_environments import (
    FrozenLake, DoorGridLarge, SokobanGate, CyclicCorridor,
)

ENVS = {
    "frozenlake": FrozenLake,
    "doorgrid": DoorGridLarge,
    "sokoban_gate": SokobanGate,
    "cyclic_corridor": CyclicCorridor,
}

ENV_MAX_STEPS = {
    "frozenlake": 50,
    "doorgrid": 80,
    "sokoban_gate": 120,
    "cyclic_corridor": 200,
}


def make_env(
    env_name: str,
    mode: str = "temporal",
    time_encoding: str = "none",
    seed: int = 42,
    sinusoidal_dim: int = 8,
) -> TemporalObsWrapper:
    """Create a wrapped environment.

    Args:
        env_name: 'frozenlake', 'doorgrid', or 'sokoban_gate'
        mode: 'static' or 'temporal'
        time_encoding: 'none', 'raw', 'sinusoidal', 'phase', 'onehot'
        seed: random seed
        sinusoidal_dim: dimension for sinusoidal encoding

    Returns:
        TemporalObsWrapper instance
    """
    if env_name not in ENVS:
        raise ValueError(f"Unknown env: {env_name}. Choose from {list(ENVS.keys())}")

    base_env = ENVS[env_name](mode=mode, seed=seed)
    max_steps = ENV_MAX_STEPS[env_name]

    return TemporalObsWrapper(
        base_env,
        time_encoding=time_encoding,
        max_steps=max_steps,
        sinusoidal_dim=sinusoidal_dim,
    )
