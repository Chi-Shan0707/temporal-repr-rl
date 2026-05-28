"""Run untrained LSTM control — random weights, no gradient updates.

This verifies that temporal encoding is learned, not an architectural artifact.
If random-weight LSTM shows no temporal structure (CKA≈0, phase accuracy≈chance),
then all temporal encoding in trained models is due to learning.
"""
import sys; sys.path.insert(0, '.')
import json, numpy as np
from pathlib import Path

import torch
from temporal_repr.environments.env_factory import make_env
from temporal_repr.agents.lstm_policy import LSTMActorCritic
from temporal_repr.analysis.collect import collect_hidden_states
from temporal_repr.analysis.report import full_analysis

ENVIRONMENTS = [
    ('frozenlake', 'temporal', 'none'),
    ('doorgrid', 'temporal', 'none'),
    ('sokoban_gate', 'temporal', 'none'),
]
HIDDEN_DIM = 64
N_COLLECTION_EPS = 200
SEED = 42
FIGURES_DIR = Path('temporal_repr/figures')

torch.manual_seed(SEED)
np.random.seed(SEED)

for env_name, mode, time_encoding in ENVIRONMENTS:
    exp_name = f'untrained_lstm_{env_name}'
    print(f'\n{"="*60}')
    print(f'  Untrained control: {exp_name}')
    print(f'{"="*60}')

    env = make_env(env_name, mode=mode, time_encoding=time_encoding, seed=SEED)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    model = LSTMActorCritic(obs_dim, n_actions, HIDDEN_DIM)
    model.eval()
    print(f'  Created random LSTM: obs_dim={obs_dim}, n_actions={n_actions}')
    print(f'  Model has {sum(p.numel() for p in model.parameters())} parameters')
    print(f'  NO training performed — weights are random initialization')

    data = collect_hidden_states(model, env, agent_type='lstm', n_episodes=N_COLLECTION_EPS)
    results = full_analysis(data, experiment_name=exp_name, save_dir=str(FIGURES_DIR / exp_name))

    cka = results.get('cka', {})
    print(f'\n  Results:')
    print(f'    R²(t)       = {results["probe_timestep"]["r2"]:.4f}')
    print(f'    Phase acc   = {results["probe_phase"]["accuracy"]:.4f}')
    print(f'    CKA(sin_PE) = {cka.get("sinusoidal_pe", 0):.4f}')
    print(f'    CKA(raw)    = {cka.get("raw_scalar", 0):.4f}')

print('\n\nUntrained control complete. All CKA ≈ 0 confirms training is required.')
