from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import signal


def _fft_features(x: np.ndarray, dt: float, n_bands: int) -> dict[str, float]:
    x_demeaned = x - np.mean(x)
    n = x_demeaned.size
    fft_vals = np.fft.rfft(x_demeaned)
    freqs = np.fft.rfftfreq(n, d=dt)

    amplitudes = np.abs(fft_vals) * (2.0 / n)
    power = np.abs(fft_vals) ** 2

    if amplitudes.size <= 1:
        f_peak = 0.0
        a_peak = 0.0
    else:
        peak_idx = 1 + int(np.argmax(amplitudes[1:]))
        f_peak = float(freqs[peak_idx])
        a_peak = float(amplitudes[peak_idx])

    fmax = float(freqs[-1]) if freqs.size else 1.0
    if fmax <= 0.0:
        fmax = 1.0

    edges = np.linspace(0.0, fmax, n_bands + 1)
    total_power = float(np.sum(power)) + 1e-12
    band_ratios = {}
    for band_idx in range(n_bands):
        left = edges[band_idx]
        right = edges[band_idx + 1]
        if band_idx == n_bands - 1:
            mask = (freqs >= left) & (freqs <= right)
        else:
            mask = (freqs >= left) & (freqs < right)
        band_energy = float(np.sum(power[mask]))
        band_ratios[f"band_{band_idx}"] = band_energy / total_power

    return {"f_peak": f_peak, "a_peak": a_peak, **band_ratios}


def _cross_probe_features(signal_matrix: np.ndarray) -> dict[str, float]:
    n_samples, n_probes = signal_matrix.shape
    if n_probes < 2:
        return {
            "adj_corr_peak_mean": 0.0,
            "adj_corr_peak_std": 0.0,
            "adj_corr_lag_mean": 0.0,
            "adj_corr_lag_std": 0.0,
        }

    peaks = []
    lags = []
    center = n_samples - 1

    for i in range(n_probes - 1):
        x = signal_matrix[:, i] - np.mean(signal_matrix[:, i])
        y = signal_matrix[:, i + 1] - np.mean(signal_matrix[:, i + 1])
        corr = signal.correlate(x, y, mode="full", method="fft")
        denom = np.sqrt(np.sum(x * x) * np.sum(y * y)) + 1e-12
        corr_norm = corr / denom

        idx = int(np.argmax(np.abs(corr_norm)))
        peaks.append(float(corr_norm[idx]))
        lags.append(float(idx - center))

    return {
        "adj_corr_peak_mean": float(np.mean(peaks)),
        "adj_corr_peak_std": float(np.std(peaks)),
        "adj_corr_lag_mean": float(np.mean(lags)),
        "adj_corr_lag_std": float(np.std(lags)),
    }


def _pod_features(signal_matrix: np.ndarray, k_modes: int) -> dict[str, float]:
    x = signal_matrix - np.mean(signal_matrix, axis=0, keepdims=True)
    _, singular_values, _ = np.linalg.svd(x, full_matrices=False)
    energies = singular_values**2
    total = float(np.sum(energies)) + 1e-12
    ratios = energies / total

    result = {}
    for idx in range(k_modes):
        value = float(ratios[idx]) if idx < ratios.size else 0.0
        result[f"pod_energy_{idx + 1}"] = value
    return result


def extract_features_from_df(df: pd.DataFrame, n_bands: int, pod_modes: int, add_pod: bool = True) -> dict[str, float]:
    time = df["time"].to_numpy(dtype=float)
    probe_cols = [col for col in df.columns if col.startswith("u_")]
    signal_matrix = df[probe_cols].to_numpy(dtype=float)

    if time.size < 4:
        raise ValueError("Not enough samples to extract features")

    dt = float(np.median(np.diff(time)))
    if dt <= 0.0:
        raise ValueError("Invalid time axis in probe data")

    features: dict[str, float] = {}

    for idx, col in enumerate(probe_cols):
        x = signal_matrix[:, idx]
        prefix = f"p{idx:03d}"

        features[f"{prefix}_mean"] = float(np.mean(x))
        features[f"{prefix}_std"] = float(np.std(x))

        fft_f = _fft_features(x, dt, n_bands=n_bands)
        features[f"{prefix}_f_peak"] = fft_f["f_peak"]
        features[f"{prefix}_a_peak"] = fft_f["a_peak"]
        for band_idx in range(n_bands):
            features[f"{prefix}_band_{band_idx}"] = fft_f[f"band_{band_idx}"]

    features.update(_cross_probe_features(signal_matrix))

    if add_pod:
        features.update(_pod_features(signal_matrix, k_modes=pod_modes))

    return features
