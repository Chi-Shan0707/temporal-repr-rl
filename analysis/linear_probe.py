"""
analysis/linear_probe.py — Linear probing of hidden states for temporal information.

Trains Ridge regression: hidden_state → timestep.
High R² means timestep is linearly decodable from representation.

Also probes for phase (classification accuracy).
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import StandardScaler


def linear_probe_timestep(
    hidden_states: np.ndarray,
    timesteps: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """Probe: can we linearly predict timestep from hidden state?

    Returns:
        r2: R² score on test set
        r2_shuffled: R² with shuffled timesteps (baseline)
        top_dims: indices of top-5 most informative dimensions (by |weight|)
        coefficients: full weight vector
    """
    X = hidden_states
    y = timesteps.astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Real probe
    model = Ridge(alpha=1.0)
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
    r2 = r2_score(y_test, y_pred)

    # Shuffled baseline
    rng = np.random.RandomState(random_state)
    y_shuffled = rng.permutation(y_train)
    model_shuf = Ridge(alpha=1.0)
    model_shuf.fit(X_train_s, y_shuffled)
    r2_shuffled = r2_score(y_test, model_shuf.predict(X_test_s))

    # Most informative dimensions
    coefs = np.abs(model.coef_)
    top_dims = np.argsort(coefs)[-5:][::-1]

    return {
        "r2": float(r2),
        "r2_shuffled": float(r2_shuffled),
        "top_dims": top_dims.tolist(),
        "coefficients": model.coef_,
        "n_samples": len(X),
    }


def linear_probe_phase(
    hidden_states: np.ndarray,
    phases: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """Probe: can we classify phase from hidden state?

    Returns:
        accuracy: classification accuracy on test set
        accuracy_shuffled: accuracy with shuffled phases
    """
    X = hidden_states
    y = phases.astype(np.int32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Real probe
    model = LogisticRegression(max_iter=1000, C=1.0, multi_class="multinomial")
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
    accuracy = accuracy_score(y_test, y_pred)

    # Shuffled baseline
    rng = np.random.RandomState(random_state)
    y_shuffled = rng.permutation(y_train)
    model_shuf = LogisticRegression(max_iter=1000, C=1.0, multi_class="multinomial")
    model_shuf.fit(X_train_s, y_shuffled)
    accuracy_shuffled = accuracy_score(y_test, model_shuf.predict(X_test_s))

    return {
        "accuracy": float(accuracy),
        "accuracy_shuffled": float(accuracy_shuffled),
        "n_classes": len(np.unique(y)),
    }
