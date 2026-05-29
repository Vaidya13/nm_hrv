"""
sqi.py
------
Signal Quality Index (SQI) computation and classification.

Provides:
  - Individual SQI metrics (flatline, clipping, QRS power, SNR,
    template correlation, RR stability, peak density)
  - Hierarchical 4-tier SQI classification (excellent / acceptable /
    borderline / poor)
  - Binary SQI accept/reject gate used by the pipeline
"""

import numpy as np
import scipy.stats
import scipy.signal
import neurokit2 as nk


# ===========================================================================
# LOW-LEVEL SIGNAL FEATURES
# ===========================================================================

def flatline_fraction(signal):
    """Fraction of consecutive sample pairs with zero change (flat signal)."""
    diffs = np.abs(np.diff(signal))
    return float(np.mean(diffs < 1e-12))


def clipping_fraction(signal):
    """
    Fraction of samples at/beyond the 0.1th and 99.9th percentiles.
    Indicates ADC saturation or hard clipping.
    """
    upper   = np.percentile(signal, 99.9)
    lower   = np.percentile(signal, 0.1)
    clipped = (signal >= upper) | (signal <= lower)
    return float(np.mean(clipped))


def signal_entropy(signal, bins=100):
    """Spectral entropy proxy via amplitude histogram."""
    hist, _ = np.histogram(signal, bins=bins, density=True)
    hist    = hist[hist > 0]
    return float(scipy.stats.entropy(hist))


# ===========================================================================
# QRS POWER RATIO  (5–15 Hz band captures QRS energy)
# ===========================================================================

def qrs_power_ratio(signal, fs):
    """
    Fraction of total PSD power contained in the 5–15 Hz QRS band.
    High ratio → dominant QRS morphology, good ECG.
    """
    freqs, psd = scipy.signal.welch(
        signal,
        fs=fs,
        nperseg=min(4096, len(signal)),
        window="hann",
    )
    total_power = np.trapz(psd, freqs)
    if total_power <= 0:
        return np.nan

    qrs_mask  = (freqs >= 5) & (freqs <= 15)
    qrs_power = np.trapz(psd[qrs_mask], freqs[qrs_mask])

    return float(qrs_power / total_power)


# ===========================================================================
# SNR ESTIMATE
# ===========================================================================

def estimate_snr(signal):
    """
    Simple SNR estimate (dB).
    Signal power = total variance; noise = residual after median smoothing.
    """
    signal_power = np.var(signal)
    noise        = signal - scipy.signal.medfilt(signal, kernel_size=11)
    noise_power  = np.var(noise)

    if noise_power <= 0:
        return np.nan

    return float(10.0 * np.log10(signal_power / noise_power))


# ===========================================================================
# TEMPLATE CORRELATION SQI  (morphological consistency)
# ===========================================================================

def template_correlation_sqi(signal, rpeaks, fs):
    """
    Compute median correlation of each beat against the median template.

    Parameters
    ----------
    signal : np.ndarray
        Preprocessed ECG signal.
    rpeaks : np.ndarray
        R-peak sample indices.
    fs : float
        Sampling frequency.

    Returns
    -------
    float
        Median beat-template correlation (0–1). NaN if < 5 valid beats.
    """
    pre  = int(0.2 * fs)
    post = int(0.4 * fs)

    beats = []
    for rp in rpeaks:
        start = rp - pre
        end   = rp + post
        if start < 0 or end >= len(signal):
            continue
        beats.append(signal[start:end])

    if len(beats) < 5:
        return np.nan

    beats    = np.array(beats)
    template = np.median(beats, axis=0)

    corrs = [np.corrcoef(b, template)[0, 1] for b in beats]
    return float(np.nanmedian(corrs))


# ===========================================================================
# RR STABILITY  (within-segment variability of successive differences)
# ===========================================================================

def rr_stability(rr_ms):
    """
    Std of successive RR differences.
    Low values → stable rhythm; high values → arrhythmia / noisy detection.
    """
    if len(rr_ms) < 5:
        return np.nan

    diffs = np.diff(rr_ms)
    return float(np.std(diffs))


def rr_coefficient_of_variation(rr_ms):
    """CV of RR intervals (dimensionless)."""
    if len(rr_ms) < 3:
        return np.nan
    mean = np.mean(rr_ms)
    if mean <= 0:
        return np.nan
    return float(np.std(rr_ms) / mean)


# ===========================================================================
# PEAK DENSITY  (instantaneous HR estimate)
# ===========================================================================

def peak_density(rpeaks, duration_sec):
    """Return estimated HR in bpm from detected peaks."""
    if duration_sec <= 0:
        return np.nan
    return float((len(rpeaks) / duration_sec) * 60.0)


# ===========================================================================
# MAIN SQI COMPUTATION
# ===========================================================================

def compute_sqi(signal, fs):
    """
    Compute the full SQI feature set for a single ECG segment.

    Parameters
    ----------
    signal : np.ndarray
        Pre-processed ECG segment (already filtered & normalised).
    fs : float
        Sampling frequency in Hz.

    Returns
    -------
    dict
        SQI feature dictionary.
    """
    out = {}

    # ---- Basic signal statistics -------------------------------------------
    out["variance"]          = float(np.var(signal))
    out["kurtosis"]          = float(scipy.stats.kurtosis(signal, fisher=False))
    out["flatline_fraction"] = flatline_fraction(signal)
    out["clipping_fraction"] = clipping_fraction(signal)
    out["signal_entropy"]    = signal_entropy(signal)
    out["qrs_power_ratio"]   = qrs_power_ratio(signal, fs)
    out["snr_db"]            = estimate_snr(signal)

    # ---- Peak detection for morphological metrics --------------------------
    try:
        cleaned = nk.ecg_clean(signal, sampling_rate=fs)
        _, info = nk.ecg_peaks(
            cleaned,
            sampling_rate=fs,
            method="pantompkins1985",
        )
        rpeaks = np.asarray(info["ECG_R_Peaks"], dtype=int)
    except Exception:
        rpeaks = np.array([], dtype=int)

    out["n_rpeaks"] = int(len(rpeaks))

    duration_sec              = len(signal) / fs
    out["peak_density_bpm"]   = peak_density(rpeaks, duration_sec)

    hr = out["peak_density_bpm"]
    out["hr_plausible"] = bool(35 <= hr <= 180) if not np.isnan(hr) else False

    # ---- RR-based metrics --------------------------------------------------
    if len(rpeaks) >= 3:
        rr_ms = np.diff(rpeaks) / fs * 1000.0
    else:
        rr_ms = np.array([])

    out["rr_stability"] = rr_stability(rr_ms)
    out["rr_cv"]        = rr_coefficient_of_variation(rr_ms)

    # ---- Morphological SQI -------------------------------------------------
    out["template_corr"] = template_correlation_sqi(signal, rpeaks, fs)

    # ---- Hierarchical SQI class & numeric score ----------------------------
    sqi_class, sqi_score  = classify_sqi(out)
    out["SQI_Class"]      = sqi_class
    out["SQI_Score"]      = sqi_score

    return out


# ===========================================================================
# HIERARCHICAL SQI CLASSIFICATION
# ===========================================================================

def classify_sqi(sqi):
    """
    Score SQI features and return a 4-tier quality label plus numeric score.

    Scoring rubric (each criterion = 1 point, max = 6):
    
    Criterion and Threshold
    QRS power ratio  > 0.40 
    Template correlation   > 0.80 
    SNR  > 10 dB 
    RR coefficient of variation < 0.20  
    Flatline fraction   < 0.05  
    HR plausible 35-180 bpm

    Returns
    -------
    (label : str, score : int)
        label ∈ {"excellent", "acceptable", "borderline", "poor"}
    """
    score = 0

    if _safe_val(sqi.get("qrs_power_ratio"), 0) > 0.40:
        score += 1
    if _safe_val(sqi.get("template_corr"), 0) > 0.80:
        score += 1
    if _safe_val(sqi.get("snr_db"), 0) > 10.0:
        score += 1
    if _safe_val(sqi.get("rr_cv"), 1) < 0.20:
        score += 1
    if _safe_val(sqi.get("flatline_fraction"), 1) < 0.05:
        score += 1
    if sqi.get("hr_plausible", False):
        score += 1

    if score >= 6:
        label = "excellent"
    elif score >= 4:
        label = "acceptable"
    elif score >= 2:
        label = "borderline"
    else:
        label = "poor"

    return label, score


def _safe_val(v, fallback):
    """Return fallback if v is None or NaN."""
    if v is None:
        return fallback
    try:
        if np.isnan(v):
            return fallback
    except TypeError:
        pass
    return v


# ===========================================================================
# BINARY ACCEPT / REJECT GATE  (used by pipeline)
# ===========================================================================

def sqi_accept(sqi, config):
    """
    Hard-threshold gate that must pass before HRV extraction.

    Returns True only when all mandatory quality checks pass.
    """
    if sqi["flatline_fraction"] > config["max_flatline_fraction"]:
        return False

    if sqi["clipping_fraction"] > config["max_clipping_fraction"]:
        return False

    if sqi["qrs_power_ratio"] < config["sqi_qrs_min"]:
        return False

    if not sqi["hr_plausible"]:
        return False

    if sqi["n_rpeaks"] < config["min_rpeaks"]:
        return False

    tc = sqi["template_corr"]
    if np.isnan(tc) or tc < config["template_corr_min"]:
        return False

    return True
