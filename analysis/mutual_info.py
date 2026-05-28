"""
analysis/mutual_info.py — Mutual information estimation between hidden states and timestep.

I(hidden_state; timestep) captures ALL forms of temporal encoding,
including nonlinear relationships that linear probes miss.

Two estimators:
  - Binning: simple, fast, lower bound
  - KSG (k-NN): more accurate for continuous variables
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from scipy.special import digamma


def estimate_mi_binning(
    hidden_states: np.ndarray,
    timesteps: np.ndarray,
    n_bins_t: int = 20,
    n_bins_h: int = 10,
) -> dict:
    """Binning-based MI estimation: I(H; T).

    Discretize both variables into bins and compute empirical MI.
    Simple but underestimates true MI (lower bound).

    Returns:
        mi: estimated mutual information in nats
        mi_normalized: mi / H(T) (fraction of temporal info captured)
        h_t: entropy of timestep distribution
    """
    n = len(timesteps)

    # Discretize timestep
    t_bins = np.linspace(timesteps.min(), timesteps.max() + 1e-8, n_bins_t + 1)
    t_discrete = np.digitize(timesteps, t_bins) - 1
    t_discrete = np.clip(t_discrete, 0, n_bins_t - 1)

    # Discretize hidden states (use first few PCs to reduce dimensionality)
    from sklearn.decomposition import PCA
    n_components = min(3, hidden_states.shape[1])
    pca = PCA(n_components=n_components)
    h_reduced = pca.fit_transform(hidden_states)

    h_discrete = np.zeros_like(h_reduced, dtype=np.int32)
    for d in range(n_components):
        bins = np.linspace(h_reduced[:, d].min(), h_reduced[:, d].max() + 1e-8, n_bins_h + 1)
        h_discrete[:, d] = np.clip(np.digitize(h_reduced[:, d], bins) - 1, 0, n_bins_h - 1)

    # Combine hidden dimensions into single index
    h_combined = h_discrete[:, 0]
    for d in range(1, n_components):
        h_combined = h_combined * n_bins_h + h_discrete[:, d]

    # Compute empirical distributions
    joint, _, _ = np.histogram2d(h_combined, t_discrete,
                                  bins=[n_bins_h ** n_components, n_bins_t])
    joint = joint / joint.sum()

    p_h = joint.sum(axis=1)
    p_t = joint.sum(axis=0)

    # MI = sum p(h,t) log(p(h,t) / (p(h)p(t)))
    mask = joint > 0
    mi = np.sum(joint[mask] * np.log(joint[mask] / (p_h[joint.nonzero()[0]] * p_t[joint.nonzero()[1]] + 1e-30)))

    # Entropy of T
    p_t_pos = p_t[p_t > 0]
    h_t = -np.sum(p_t_pos * np.log(p_t_pos))

    return {
        "mi": float(mi),
        "mi_normalized": float(mi / h_t) if h_t > 0 else 0.0,
        "h_t": float(h_t),
        "n_bins": n_bins_t,
    }


def estimate_mi_ksg(
    hidden_states: np.ndarray,
    timesteps: np.ndarray,
    k: int = 5,
    subsample: int = 5000,
) -> dict:
    """KSG (Kraskov-Stögbauer-Grassberger) k-NN MI estimator.

    More accurate for continuous variables. Uses k-nearest-neighbor distances.

    Args:
        hidden_states: (n, d)
        timesteps: (n,)
        k: number of neighbors
        subsample: max samples (KSG is O(n²), subsample for large datasets)

    Returns:
        mi: estimated MI in nats
    """
    n = len(timesteps)
    if n > subsample:
        rng = np.random.RandomState(42)
        idx = rng.choice(n, subsample, replace=False)
        H = hidden_states[idx]
        T = timesteps[idx].reshape(-1, 1).astype(np.float64)
    else:
        H = hidden_states.astype(np.float64)
        T = timesteps.reshape(-1, 1).astype(np.float64)

    n = len(T)

    # Normalize
    H = (H - H.mean(axis=0)) / (H.std(axis=0) + 1e-8)
    T = (T - T.mean(axis=0)) / (T.std(axis=0) + 1e-8)

    # Joint space
    HT = np.hstack([H, T])

    # Build KD-trees
    tree_ht = cKDTree(HT)
    tree_h = cKDTree(H)
    tree_t = cKDTree(T)

    # KSG estimator (algorithm 1)
    eps, _ = tree_ht.query(HT, k=k+1)  # k+1 because first is self
    eps = eps[:, -1]  # distance to k-th neighbor

    # Count neighbors within eps in marginal spaces
    n_h = np.array([len(tree_h.query_ball_point(H[i], eps[i])) - 1 for i in range(n)])
    n_t = np.array([len(tree_t.query_ball_point(T[i], eps[i])) - 1 for i in range(n)])

    mi = digamma(k) - np.mean(digamma(n_h + 1) + digamma(n_t + 1))

    return {
        "mi": float(max(mi, 0.0)),
        "k": k,
        "n_samples": n,
    }
