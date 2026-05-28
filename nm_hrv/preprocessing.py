"""
preprocessing.py
----------------
ECG signal preprocessing:
  - Polynomial detrending
  - Bandpass filter (0.5-40 Hz)
  - Notch filter (50/60 Hz)
  - Robust z-score normalisation
"""

import numpy as np
import neurokit2 as nk

from scipy.signal import butter, filtfilt, iirnotch


# ---------------------------------------------------------------------------
# Default config (overridden by YAML at runtime)
# ---------------------------------------------------------------------------

DEFAULT_PREPROCESS_CFG = {
    "hp_cutoff":    0.5,    # Hz  – high-pass corner
    "lp_cutoff":    40.0,   # Hz  – low-pass corner
    "notch_freq":   50.0,   # Hz  – mains frequency (use 60 for US)
    "filter_order": 2,      # Butterworth order
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def robust_zscore(x):
    """
    Normalise using median and MAD (robust to outliers).
    Returns z-scored signal preserving shape.
    """
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))

    if mad < 1e-12:
        return x - med

    return (x - med) / (1.4826 * mad)   # 1.4826 makes MAD consistent with σ


# ---------------------------------------------------------------------------
# Main preprocessor
# ---------------------------------------------------------------------------

def preprocess_ecg(ecg, fs, cfg=None):
    """
    Standard ECG preprocessing pipeline.

    Steps
    -----
    1. Polynomial baseline detrend (order 1)
    2. Bandpass 0.5–40 Hz (Butterworth)
    3. Notch at mains frequency (50 or 60 Hz)
    4. Robust z-score normalisation

    Parameters
    ----------
    ecg : array_like
        Raw ECG signal (any amplitude scale).
    fs : float
        Sampling frequency in Hz.
    cfg : dict, optional
        Override DEFAULT_PREPROCESS_CFG keys.

    Returns
    -------
    ecg_clean : np.ndarray
        Preprocessed, normalised ECG.
    """
    if cfg is None:
        cfg = DEFAULT_PREPROCESS_CFG

    ecg = np.asarray(ecg, dtype=float)
    fs  = float(fs)

    if len(ecg) < 3:
        raise ValueError(f"ECG segment too short: {len(ecg)} samples")

    # ---- 1. Baseline wander removal ----------------------------------------
    ecg = nk.signal_detrend(ecg, method="polynomial", order=1)

    # ---- 2. Bandpass 0.5–40 Hz ------------------------------------------------
    nyq = fs / 2.0
    hp  = cfg["hp_cutoff"] / nyq
    lp  = cfg["lp_cutoff"] / nyq

    # Guard against impossible filter specs at low sampling rates
    hp = np.clip(hp, 1e-4, 0.99)
    lp = np.clip(lp, hp + 1e-4, 0.99)

    b, a = butter(cfg["filter_order"], [hp, lp], btype="band")
    ecg  = filtfilt(b, a, ecg)

    # ---- 3. Notch for mains interference ------------------------------------
    w0       = cfg["notch_freq"] / nyq
    w0       = np.clip(w0, 1e-4, 0.999)
    b_n, a_n = iirnotch(w0, Q=30)
    ecg      = filtfilt(b_n, a_n, ecg)

    # ---- 4. Robust normalisation --------------------------------------------
    ecg = robust_zscore(ecg)

    return ecg