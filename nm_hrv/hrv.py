"""
hrv.py
------
HRV feature extraction — Kubios / Task Force (1996) standard.

Frequency-domain pipeline
--------------------------
  1. Cubic spline interpolation onto a regular 4 Hz time axis
  2. Smoothness-priors detrending  (Tarvainen et al. 2002)
  3. Welch PSD — 8 overlapping windows, 50 % overlap, Hamming taper
  4. Power integration over standard bands:
       VLF  0.003-0.04 Hz
       LF   0.04 -0.15 Hz
       HF   0.15 -0.40 Hz

References
----------
Task Force of ESC & NASPE. Circulation. 1996;93:1043-1065.
Tarvainen MP et al. IEEE Trans Biomed Eng. 2002;49(2):172-175.
Kubios HRV Scientific Edition documentation, 2024.
"""

import numpy as np

from scipy.signal        import welch
from scipy.interpolate   import CubicSpline

from .utils import smoothness_priors_detrend


# ---------------------------------------------------------------------------
# Band definitions  (Task Force 1996)
# ---------------------------------------------------------------------------

FREQ_BANDS = {
    "VLF": (0.003, 0.04),
    "LF":  (0.04,  0.15),
    "HF":  (0.15,  0.40),
}

RR_INTERP_FS = 4.0          # Hz – standard resampling rate (Kubios default)
MIN_NN_FOR_HRV = 30         # minimum NN count to attempt any HRV


# ===========================================================================
# FREQUENCY-DOMAIN HRV  (4 Hz cubic spline + smoothness priors + Welch)
# ===========================================================================

def compute_frequency_hrv(nn_ms, lambda_val=300, nperseg=256, noverlap=128, window="hann"):
    """
    Compute VLF, LF, HF power using the Kubios-standard pipeline.

    Parameters
    ----------
    nn_ms : np.ndarray
        Artefact-corrected NN intervals (ms), beat-indexed (not evenly spaced).
    lambda_val : float
        Smoothness-priors regularisation (Kubios default = 300).

    Returns
    -------
    dict
        Frequency-domain HRV metrics, or None on failure.
    """
    nn = np.asarray(nn_ms, dtype=float)

    # Beat timestamps (seconds) derived from cumulative NN sum
    t_beats = np.concatenate([[0.0], np.cumsum(nn[:-1])]) / 1000.0
    t_total = t_beats[-1] + nn[-1] / 1000.0

    # ---- Step 1: cubic spline → 4 Hz regular grid -------------------------
    t_grid = np.arange(0, t_total, 1.0 / RR_INTERP_FS)

    if len(t_grid) < 8:
        return None

    try:
        cs        = CubicSpline(t_beats, nn, extrapolate=False)
        nn_interp = cs(t_grid)
    except Exception:
        return None

    # Handle NaN at edges from extrapolation guard
    valid = ~np.isnan(nn_interp)
    if valid.sum() < 8:
        return None
    nn_interp = nn_interp[valid]

    # ---- Step 2: smoothness-priors detrending (Tarvainen 2002) -------------
    nn_detrended = smoothness_priors_detrend(nn_interp, lambda_val=lambda_val)

    # ---- Step 3: Welch PSD — configurable, defaults to hann/256/128 --------
    # nperseg=256, noverlap=128 with fs=4 Hz:
    #   freq resolution = 4/256 = 0.0156 Hz
    # Cap nperseg to segment length so we never crash on short windows.
    _nperseg  = min(nperseg, len(nn_detrended))
    _nperseg  = max(_nperseg, 16)
    _noverlap = min(noverlap, _nperseg - 1)

    freqs, psd = welch(
        nn_detrended,
        fs=RR_INTERP_FS,
        nperseg=_nperseg,
        noverlap=_noverlap,
        window=window,
        scaling="density",
    )

    # ---- Step 4: integrate over bands -------------------------------------
    def band_power(lo, hi):
        mask = (freqs >= lo) & (freqs <= hi)
        if mask.sum() == 0:
            return np.nan
        return float(np.trapz(psd[mask], freqs[mask]))

    vlf = band_power(*FREQ_BANDS["VLF"])
    lf  = band_power(*FREQ_BANDS["LF"])
    hf  = band_power(*FREQ_BANDS["HF"])

    total = (vlf if not np.isnan(vlf) else 0) + \
            (lf  if not np.isnan(lf)  else 0) + \
            (hf  if not np.isnan(hf)  else 0)

    lf_hf  = (lf / hf)  if (not np.isnan(lf) and not np.isnan(hf) and hf > 0) else np.nan
    lf_nu  = (lf / (lf + hf) * 100) if (not np.isnan(lf) and not np.isnan(hf) and (lf + hf) > 0) else np.nan
    hf_nu  = (hf / (lf + hf) * 100) if (not np.isnan(lf) and not np.isnan(hf) and (lf + hf) > 0) else np.nan

    return {
        "VLF":       vlf,
        "LF":        lf,
        "HF":        hf,
        "LF_HF":     lf_hf,
        "LF_nu":     lf_nu,
        "HF_nu":     hf_nu,
        "TP":        float(total),
        "logLF":     float(np.log(lf  + 1e-10)) if not np.isnan(lf)  else np.nan,
        "logHF":     float(np.log(hf  + 1e-10)) if not np.isnan(hf)  else np.nan,
        "logVLF":    float(np.log(vlf + 1e-10)) if not np.isnan(vlf) else np.nan,
    }


# ===========================================================================
# NONLINEAR HRV  (Poincaré SD1/SD2 + approximate entropy proxy)
# ===========================================================================

def compute_nonlinear_hrv(nn_ms):
    """
    Compute nonlinear HRV indices from NN intervals.

    SD1 : short-term variability (beat-to-beat, parasympathetic proxy)
    SD2 : long-term variability
    SD1_SD2 : SD1/SD2 ratio (sympathovagal balance proxy)
    SD1_c : HR-corrected SD1 (SD1 / MeanNN) — dimensionless, cross-cohort comparable
    """
    nn = np.asarray(nn_ms, dtype=float)

    if len(nn) < 4:
        return {}

    mean_nn = float(np.mean(nn))
    diff_nn = np.diff(nn)

    sd1 = float(np.sqrt(0.5) * np.std(diff_nn,  ddof=1))
    sd2 = float(np.sqrt(2.0 * np.var(nn, ddof=1) - 0.5 * np.var(diff_nn, ddof=1)))
    sd2 = max(sd2, 0.0)   # guard numerical noise

    sd1_sd2 = (sd1 / sd2) if sd2 > 0 else np.nan

    return {
        "SD1":          sd1,
        "SD2":          sd2,
        "SD1_SD2":      sd1_sd2,
        "SD1_sq":       sd1 ** 2,
        "SD2_sq":       sd2 ** 2,
        "SD1_c":        float(sd1 / mean_nn) if mean_nn > 0 else np.nan,  # HR-corrected
        "SampEn_proxy": float(sd1_sd2) if not np.isnan(sd1_sd2) else np.nan,
    }


# ===========================================================================
# TIME-DOMAIN HRV  (Task Force 1996)
# ===========================================================================

def compute_time_domain_hrv(nn_ms):
    """
    Compute standard time-domain HRV indices.

    Includes:
      - Raw indices (Task Force 1996)
      - Log-transformed indices (lnRMSSD, lnSDNN) — approximately normal,
        required for parametric normative modelling (GAMLSS / quantile regression)
      - HR-corrected indices (RMSSD_c, SDNN_c, SD1_c) — dimensionless,
        removes the mathematical confound between HR and HRV magnitude.
        Essential when comparing children vs adults or across cohorts with
        different mean heart rates.
        Method: Sacha & Pluta (2005); Gąsior et al. (2018 Front Physiol)

    Returns
    -------
    dict
    """
    nn = np.asarray(nn_ms, dtype=float)

    if len(nn) < 4:
        return {}

    mean_nn   = float(np.mean(nn))
    sdnn      = float(np.std(nn, ddof=1))
    diff_nn   = np.diff(nn)
    rmssd     = float(np.sqrt(np.mean(diff_nn ** 2)))
    pnn50     = float(np.mean(np.abs(diff_nn) > 50.0) * 100.0)
    pnn20     = float(np.mean(np.abs(diff_nn) > 20.0) * 100.0)
    mean_hr   = 60000.0 / mean_nn
    hr_series = 60000.0 / nn
    sd_hr     = float(np.std(hr_series, ddof=1))

    # Log transforms — stabilise right-skewed distributions for normative modelling
    ln_rmssd = float(np.log(rmssd + 1e-10))
    ln_sdnn  = float(np.log(sdnn  + 1e-10))

    # HR-corrected (normalised by mean NN) — removes HR-HRV mathematical confound
    rmssd_c = float(rmssd / mean_nn) if mean_nn > 0 else np.nan
    sdnn_c  = float(sdnn  / mean_nn) if mean_nn > 0 else np.nan

    return {
        "MeanNN_ms":   mean_nn,
        "SDNN_ms":     sdnn,
        "RMSSD_ms":    rmssd,
        "lnRMSSD":     ln_rmssd,
        "lnSDNN":      ln_sdnn,
        "RMSSD_c":     rmssd_c,   # RMSSD / MeanNN  (dimensionless)
        "SDNN_c":      sdnn_c,    # SDNN  / MeanNN  (dimensionless)
        "pNN50":       pnn50,
        "pNN20":       pnn20,
        "MeanHR_bpm":  mean_hr,
        "SDHR_bpm":    sd_hr,
    }


# ===========================================================================
# MAIN HRV ENTRY POINT
# ===========================================================================

def compute_hrv(rr_ms, min_nn=MIN_NN_FOR_HRV, lambda_val=300,
                nperseg=256, noverlap=128, psd_window="hann"):
    """
    Compute the full HRV feature set from an artefact-corrected NN series.

    Parameters
    ----------
    rr_ms : array_like
        Artefact-corrected NN intervals (ms) — output of detect_rr().
    min_nn : int
        Minimum number of NN intervals required (default 30).
        For a 5-minute window at 60-80 bpm this is 300-400 beats.
        Setting min_rr_for_hrv: 150 in config provides a sensible stricter
        guard for frequency-domain validity (need ≥ 150 NN for PSD resolution).
    lambda_val : float
        Smoothness-priors lambda for frequency-domain detrending.
        Kubios default = 300 for 5-min; use 500 for children or long recordings.

    Returns
    -------
    dict or None
        Full HRV feature dictionary. None if too few intervals or failure.

    New columns vs v1
    -----------------
    lnRMSSD, lnSDNN      — log-transformed for normative modelling
    RMSSD_c, SDNN_c      — HR-corrected (÷ MeanNN), cross-cohort comparable
    SD1_c                — HR-corrected SD1
    pNN20                — pNN20 added (useful for children / high HR)
    SDHR_bpm             — SD of HR series
    """
    if rr_ms is None:
        return None

    nn_ms = np.asarray(rr_ms, dtype=float)
    nn_ms = nn_ms[~np.isnan(nn_ms)]

    if len(nn_ms) < min_nn:
        return None

    # ---- Time domain -------------------------------------------------------
    td = compute_time_domain_hrv(nn_ms)
    if not td:
        return None

    # ---- Frequency domain --------------------------------------------------
    fd = compute_frequency_hrv(
        nn_ms,
        lambda_val = lambda_val,
        nperseg    = nperseg,
        noverlap   = noverlap,
        window     = psd_window,
    )

    # ---- Nonlinear ---------------------------------------------------------
    nl = compute_nonlinear_hrv(nn_ms)

    # ---- Assemble output ---------------------------------------------------
    results = {
        # Metadata
        "NN_Count": int(len(nn_ms)),

        # Time domain (includes lnRMSSD, lnSDNN, RMSSD_c, SDNN_c)
        **td,

        # Nonlinear (includes SD1_c)
        **nl,
    }

    # Frequency domain (may be None if segment too short for PSD)
    if fd is not None:
        results.update(fd)
    else:
        for key in ["VLF", "LF", "HF", "LF_HF", "LF_nu", "HF_nu",
                    "TP", "logLF", "logHF", "logVLF"]:
            results[key] = np.nan

    return results