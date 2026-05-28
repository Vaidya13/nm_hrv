"""
rr.py
-----
RR / NN interval processing:

  1. R-peak detection via neurokit2
  2. Artifact classification using adaptive Lipponen & Tarvainen (2019)
     thresholds — automatically adjusts to variable HR, sleep, children,
     arrhythmia burden without per-cohort config changes.
  3. Cubic spline interpolation of artifact beats
  4. Outputs clean NN series ready for HRV computation

Artifact categories
-------------------
  0  normal
  1  too_short      below global plausibility limit
  2  too_long       above global plausibility limit
  3  local_outlier  generic deviation exceeding adaptive threshold
  4  missed_beat    RR >> local median (gap implies skipped beat)
  5  extra_beat     RR << local median (implies ectopic/double detection)
  6  arrhythmia     sustained successive-difference irregularity

References
----------
Lipponen JA, Tarvainen MP. J Med Eng Technol. 2019;43(3):173-181.
Task Force. Circulation. 1996;93:1043-1065.
"""

import numpy as np
import neurokit2 as nk

from scipy.interpolate import CubicSpline

from .utils import get_age_adaptive_rr_bounds


# ===========================================================================
# ARTIFACT CLASSIFICATION
# ===========================================================================

def classify_rr_artifacts(rr_ms, cfg, age=np.nan):
    """
    Label every RR interval with an artifact code.

    Uses adaptive, HR-sensitive thresholds inspired by Lipponen & Tarvainen
    (2019): the deviation threshold adapts to local MAD, which automatically
    accommodates high HRV (sleep, children, arrhythmia) without manual tuning.

    Parameters
    ----------
    rr_ms : array_like
        RR intervals in milliseconds.
    cfg : dict
        Pipeline config dict (keys: enable_age_adaptive_rules, rr_min_ms,
        rr_max_ms, local_window_beats, missed_beat_gap_factor,
        extra_beat_gap_factor, local_dev_thresh).
    age : float, optional
        Subject age in years (for age-adaptive physiological limits).

    Returns
    -------
    flags : np.ndarray of int
        Same length as rr_ms, with artifact codes 0–6.
    """
    rr    = np.asarray(rr_ms, dtype=float)
    n     = len(rr)
    flags = np.zeros(n, dtype=int)

    # ---- Global plausibility limits (hard) ---------------------------------
    if cfg.get("enable_age_adaptive_rules", False):
        rr_min_ms, rr_max_ms = get_age_adaptive_rr_bounds(age)
    else:
        rr_min_ms = cfg["rr_min_ms"]
        rr_max_ms = cfg["rr_max_ms"]

    flags[rr < rr_min_ms] = 1
    flags[rr > rr_max_ms] = 2

    # ---- Local adaptive rule (Lipponen & Tarvainen style) ------------------
    w    = cfg.get("local_window_beats", 11)
    half = w // 2

    for i in range(n):
        if flags[i] != 0:
            continue

        lo = max(0, i - half)
        hi = min(n, i + half + 1)

        local_rr = rr[lo:hi]
        med      = np.nanmedian(local_rr)

        if med <= 0:
            continue

        mad = np.nanmedian(np.abs(local_rr - med))
        mad = max(mad, 1e-6)

        # Adaptive threshold: never less than 20 % of local median
        # (mirrors Kubios "medium" threshold adjusted for HR level)
        adaptive_thresh = max(
            cfg.get("local_dev_thresh", 0.20),
            3.0 * mad / med,
        )

        rel = (rr[i] - med) / med

        # Missed beat: RR >> local median
        if rel > (cfg["missed_beat_gap_factor"] - 1):
            flags[i] = 4
            continue

        # Extra beat: RR << local median
        if rel < -(1 - cfg["extra_beat_gap_factor"]):
            flags[i] = 5
            continue

        # Generic local outlier
        if abs(rel) > adaptive_thresh:
            flags[i] = 3
            continue

    # ---- Arrhythmia-like: sustained successive-difference irregularity ------
    if n >= 5:
        diffs       = np.diff(rr)
        local_scale = np.nanmedian(np.abs(diffs - np.nanmedian(diffs)))
        local_scale = max(local_scale, 1e-6)
        zdiff       = np.abs(diffs) / (1.4826 * local_scale)

        for i in range(1, len(zdiff)):
            if zdiff[i - 1] > 3.0 and zdiff[i] > 3.0:
                if flags[i] == 0:
                    flags[i] = 6

    return flags


# ===========================================================================
# CUBIC SPLINE INTERPOLATION  (Kubios standard correction)
# ===========================================================================

def interpolate_rr(rr_ms, flags, cfg):
    """
    Replace artifact beats with cubic spline interpolated values.

    Parameters
    ----------
    rr_ms : array_like
        Original RR series (ms).
    flags : np.ndarray
        Artifact flags (0 = normal).
    cfg : dict
        Must contain 'max_artifact_fraction'.

    Returns
    -------
    rr_interp : np.ndarray or None
        Corrected NN series (ms). None if artifact burden is too high.
    artifact_fraction : float
        Fraction of beats classified as artefact.
    """
    rr       = np.asarray(rr_ms, dtype=float)
    rr_clean = rr.copy()
    rr_clean[flags != 0] = np.nan

    valid             = ~np.isnan(rr_clean)
    artifact_fraction = float(1.0 - np.mean(valid))

    if artifact_fraction > cfg["max_artifact_fraction"]:
        return None, artifact_fraction

    x       = np.arange(len(rr))
    x_valid = x[valid]
    y_valid = rr_clean[valid]

    if len(x_valid) < 2:
        return None, artifact_fraction

    if len(x_valid) >= 4:
        cs         = CubicSpline(x_valid, y_valid, extrapolate=True)
        rr_interp  = cs(x)
    else:
        rr_interp = np.interp(x, x_valid, y_valid)

    return rr_interp, artifact_fraction


# ===========================================================================
# R-PEAK DETECTION + FULL RR PIPELINE
# ===========================================================================

def detect_rr(ecg, fs, config, age=np.nan):
    """
    Full pipeline: ECG → R-peaks → RR intervals → artifact correction → NN.

    Parameters
    ----------
    ecg : np.ndarray
        Pre-processed ECG segment (normalised).
    fs : float
        Sampling frequency in Hz.
    config : dict
        Pipeline configuration dictionary.
    age : float, optional
        Subject age for age-adaptive limits.

    Returns
    -------
    dict or None
        Keys: rpeaks, rr_ms_raw, rr_flags, rr_ms_nn, artifact_fraction,
              n_beats, n_artifacts.
        Returns None if fewer than 3 peaks detected.
    """
    ecg_cleaned = nk.ecg_clean(ecg, sampling_rate=fs)

    _, info = nk.ecg_peaks(
        ecg_cleaned,
        sampling_rate=fs,
        method=config.get("ecg_peaks_method", "pantompkins1985"),
    )

    rpeaks = np.asarray(info["ECG_R_Peaks"], dtype=int)

    if len(rpeaks) < 3:
        return None

    rr_ms = np.diff(rpeaks) / fs * 1000.0

    flags = classify_rr_artifacts(rr_ms, config, age=age)

    rr_interp, art_frac = interpolate_rr(rr_ms, flags, config)

    return {
        "rpeaks":            rpeaks,
        "rr_ms_raw":         rr_ms,
        "rr_flags":          flags,
        "rr_ms_nn":          rr_interp,
        "artifact_fraction": art_frac,
        "n_beats":           int(len(rr_ms)),
        "n_artifacts":       int(np.sum(flags != 0)),
    }