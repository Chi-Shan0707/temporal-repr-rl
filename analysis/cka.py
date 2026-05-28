from __future__ import annotations
"""
analysis/cka.py — Centered Kernel Alignment (CKA) for comparing representations.

Compares learned LSTM hidden states against engineered temporal encodings:
  - raw scalar: t / max_t
  - sinusoidal PE: Transformer-style multi-frequency
  - phase sin/cos: circular encoding
  - one-hot phase

CKA ≈ 1.0 means representations are similar (up to orthogonal rotation).
"""

from pathlib import Path

import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from temporal_repr.environments.wrappers import sinusoidal_encode


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA (Kornblith et al., 2019).

    Args:
        X: (n, d_x)
        Y: (n, d_y)

    Returns:
        CKA similarity in [0, 1]
    """
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)

    # HSIC
    def hsic(A, B):
        # Unbiased estimator
        n = A.shape[0]
        if n < 4:
            return 0.0
        # Linear kernel
        K = A @ A.T
        L = B @ B.T
        # Zero out diagonal for unbiased
        np.fill_diagonal(K, 0)
        np.fill_diagonal(L, 0)
        return np.sum(K * L) / (n * (n - 1))

    hsic_xy = hsic(X, Y)
    hsic_xx = hsic(X, X)
    hsic_yy = hsic(Y, Y)

    if hsic_xx <= 0 or hsic_yy <= 0:
        return 0.0

    return float(hsic_xy / (np.sqrt(hsic_xx * hsic_yy) + 1e-10))


def generate_engineered_encodings(
    timesteps: np.ndarray,
    phases: np.ndarray,
    period: int,
    max_t: int,
    hidden_dim: int = 64,
) -> dict[str, np.ndarray]:
    """Generate all engineered temporal encodings for comparison."""
    n = len(timesteps)
    encodings = {}

    # 1. Raw scalar
    encodings["raw_scalar"] = (timesteps / max_t).reshape(-1, 1)

    # 2. Sinusoidal PE (Transformer-style)
    sin_enc = np.array([sinusoidal_encode(int(t), dim=min(16, hidden_dim)) for t in timesteps])
    encodings["sinusoidal_pe"] = sin_enc

    # 3. Phase sin/cos (circular)
    phase_enc = np.stack([
        np.sin(2 * np.pi * phases / period),
        np.cos(2 * np.pi * phases / period),
    ], axis=1)
    encodings["phase_sin_cos"] = phase_enc

    # 4. One-hot phase
    oh = np.zeros((n, period), dtype=np.float32)
    oh[np.arange(n), phases % period] = 1.0
    encodings["onehot_phase"] = oh

    # 5. Remaining time (Pardo et al.)
    encodings["remaining_time"] = ((max_t - timesteps) / max_t).reshape(-1, 1)

    return encodings


def compare_encodings(
    hidden_states: np.ndarray,
    timesteps: np.ndarray,
    phases: np.ndarray,
    period: int,
    max_t: int = 100,
    max_samples: int = 2000,
) -> dict[str, float]:
    """Compare learned hidden states against all engineered encodings via CKA.

    Returns dict mapping encoding name → CKA score.
    """
    n = len(timesteps)
    if n > max_samples:
        rng = np.random.RandomState(42)
        idx = rng.choice(n, max_samples, replace=False)
        hidden_states = hidden_states[idx]
        timesteps = timesteps[idx]
        phases = phases[idx]

    encodings = generate_engineered_encodings(timesteps, phases, period, max_t)

    results = {}
    for name, enc in encodings.items():
        cka_score = linear_cka(hidden_states, enc)
        results[name] = cka_score

    return results
