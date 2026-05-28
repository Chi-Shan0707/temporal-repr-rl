"""
analysis/fourier.py — Fourier analysis of hidden state trajectories.

THE KEY NOVELTY: Analyzes the frequency spectrum of RNN hidden states
to determine what temporal frequencies the agent has learned.

For each hidden dimension:
  1. Sort by timestep to get a time-series
  2. Detrend (subtract mean)
  3. Apply Hanning window
  4. FFT → power spectrum
  5. Average across dimensions

Key outputs:
  - Power at frequencies matching environment period
  - Dominant frequencies
  - Spectral centroid
"""

from __future__ import annotations

import numpy as np
from scipy import signal as sp_signal
from scipy.fft import rfft, rfftfreq


def fourier_analysis(
    hidden_states: np.ndarray,
    timesteps: np.ndarray,
    period: int | None = None,
    max_freq: float = 0.5,
) -> dict:
    """Fourier analysis of hidden state trajectories.

    Args:
        hidden_states: (n_steps, hidden_dim)
        timesteps: (n_steps,) absolute timestep
        period: environment period (if known)
        max_freq: maximum frequency to report (cycles/step)

    Returns:
        freqs: frequency array
        avg_power: average power spectrum across dimensions
        per_dim_power: (hidden_dim, freq_bins) per-dimension spectra
        dominant_freqs: top-k dominant frequencies
        env_period_power: power at f = 1/period (if period given)
        spectral_centroid: weighted average frequency
        top_freq_dims: which hidden dimensions carry the most temporal info
    """
    n_steps, hidden_dim = hidden_states.shape

    # Detect episode boundaries directly from timestep resets (don't sort — keep episode order)
    episode_starts = [0]
    for i in range(1, len(timesteps)):
        if timesteps[i] <= timesteps[i - 1]:
            episode_starts.append(i)
    episode_starts.append(len(timesteps))

    # Minimum segment length for FFT
    min_seg = max(4, min(8, n_steps))

    # Compute FFT on each episode segment, then average
    all_spectra = []

    for e in range(len(episode_starts) - 1):
        start = episode_starts[e]
        end = episode_starts[e + 1]
        segment = hidden_states[start:end]

        if len(segment) < min_seg:
            continue

        n = len(segment)
        freqs = rfftfreq(n, d=1.0)  # frequency in cycles/step
        window = np.hanning(n)

        for dim in range(hidden_dim):
            ts = segment[:, dim]
            ts = ts - ts.mean()  # detrend
            ts = ts * window
            spectrum = np.abs(rfft(ts)) ** 2 / n
            all_spectra.append((freqs, spectrum))

    if not all_spectra:
        return {"error": "No valid episode segments for FFT"}

    # Interpolate all spectra to a common frequency grid
    freq_min = all_spectra[0][0][1]  # skip DC
    freq_max = max_freq
    n_freq_bins = 256
    common_freqs = np.linspace(freq_min, freq_max, n_freq_bins)

    interpolated = []
    for freqs_raw, spectrum_raw in all_spectra:
        mask = (freqs_raw >= freq_min) & (freqs_raw <= freq_max)
        if mask.sum() < 2:
            continue
        f = freqs_raw[mask]
        s = spectrum_raw[mask]
        interp = np.interp(common_freqs, f, s)
        interpolated.append(interp)

    avg_power = np.mean(interpolated, axis=0)

    # Find dominant frequencies
    peaks, properties = sp_signal.find_peaks(avg_power, height=np.percentile(avg_power, 75))
    if len(peaks) > 0:
        peak_heights = avg_power[peaks]
        top_k = min(5, len(peaks))
        top_indices = peaks[np.argsort(peak_heights)[-top_k:]][::-1]
        dominant_freqs = common_freqs[top_indices].tolist()
        dominant_powers = avg_power[top_indices].tolist()
    else:
        dominant_freqs = []
        dominant_powers = []

    # Power at environment period frequency
    env_period_power = None
    if period is not None:
        target_freq = 1.0 / period
        if freq_min <= target_freq <= freq_max:
            idx = np.argmin(np.abs(common_freqs - target_freq))
            # Average power in a small window around target
            window = 3
            lo = max(0, idx - window)
            hi = min(len(avg_power), idx + window + 1)
            env_period_power = float(np.mean(avg_power[lo:hi]))

    # Spectral centroid
    total_power = avg_power.sum()
    if total_power > 0:
        spectral_centroid = float(np.sum(common_freqs * avg_power) / total_power)
    else:
        spectral_centroid = 0.0

    # Top dimensions by total power (most temporally active)
    dim_total_power = []
    for dim in range(hidden_dim):
        # Collect spectra for this dimension across episodes
        dim_spectra = []
        for e_idx in range(0, len(all_spectra), hidden_dim):
            if e_idx + dim < len(all_spectra):
                freqs_raw, spectrum_raw = all_spectra[e_idx + dim]
                mask = (freqs_raw >= freq_min) & (freqs_raw <= freq_max)
                if mask.sum() >= 2:
                    dim_spectra.append(np.interp(common_freqs, freqs_raw[mask], spectrum_raw[mask]))
        if dim_spectra:
            dim_total_power.append(np.mean(dim_spectra))
        else:
            dim_total_power.append(0.0)

    top_freq_dims = np.argsort(dim_total_power)[-5:][::-1].tolist()

    return {
        "freqs": common_freqs,
        "avg_power": avg_power,
        "dominant_freqs": dominant_freqs,
        "dominant_powers": dominant_powers,
        "env_period_power": env_period_power,
        "env_period_freq": 1.0 / period if period else None,
        "spectral_centroid": spectral_centroid,
        "top_freq_dims": top_freq_dims,
        "n_episodes_analyzed": len(episode_starts) - 1,
    }
