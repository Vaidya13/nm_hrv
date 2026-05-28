"""
summaries.py
------------
Record-level and cohort-level HRV summary statistics.
"""

import warnings
import numpy as np
import pandas as pd


_ARRAY_COLS = {"ecg", "rr_ms_raw", "rr_ms_nn", "rr_flags"}

# All quality labels the pipeline can emit
_QUALITY_LABELS = [
    "good",
    "good_low_sqi",        # HRV computed but SQI below threshold
    "poor_sqi",            # gate mode only - HRV not computed
    "too_many_artifacts",
    "peak_detection_failed",
    "hrv_failed",
]

# Labels for which HRV was actually computed
_HRV_QUALITY_LABELS = {"good", "good_low_sqi"}

# HRV columns to average over HRV windows in per-record summary
_HRV_SUMMARY_COLS = [
    "MeanHR_bpm", "RMSSD_ms", "SDNN_ms",
    "lnRMSSD", "lnSDNN", "RMSSD_c", "SDNN_c",
    "LF", "HF", "LF_HF", "LF_nu", "HF_nu", "VLF", "TP",
    "SD1", "SD2", "SD1_c",
    "HRV_Confidence",
]


def summarize_hrv_record(df_hrv):
    """
    Summarise all windows for one participant into a single row.

    HRV metrics are averaged over all windows where HRV was computed
    (Quality == "good" OR "good_low_sqi"), so dry-electrode data is
    included even when SQI thresholds were not met.

    Parameters
    ----------
    df_hrv : pd.DataFrame
        Output of process_record() - one row per window.

    Returns
    -------
    dict or None
    """
    if df_hrv is None or len(df_hrv) == 0:
        return None

    total      = len(df_hrv)
    good_mask  = df_hrv["Quality"] == "good"
    hrv_mask   = df_hrv["Quality"].isin(_HRV_QUALITY_LABELS)
    n_good     = int(good_mask.sum())
    n_hrv      = int(hrv_mask.sum())

    summary = {
        "n_windows":        total,
        "n_good":           n_good,
        "n_hrv_computed":   n_hrv,           # good + good_low_sqi
        "usable_fraction":  float(n_good / total),
        "hrv_fraction":     float(n_hrv  / total),   # fraction with HRV

        # Per-label proportions
        **{
            f"prop_{lbl}": float(np.mean(df_hrv["Quality"] == lbl))
            for lbl in _QUALITY_LABELS
        },

        "dominant_failure": _dominant_failure(df_hrv),
    }

    # Artifact burden
    if "ArtifactFraction" in df_hrv.columns:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            summary["ArtifactFraction_mean"]   = _safe_mean(df_hrv["ArtifactFraction"])
            summary["ArtifactFraction_median"] = _safe_median(df_hrv["ArtifactFraction"])

    # HRV metrics — mean over all windows where HRV was computed
    df_hrv_rows = df_hrv[hrv_mask]

    for col in _HRV_SUMMARY_COLS:
        if col in df_hrv_rows.columns:
            summary[col] = _safe_mean(df_hrv_rows[col])
        else:
            summary[col] = np.nan

    # SQI class distribution
    if "SQI_Class" in df_hrv.columns:
        for cls in ("excellent", "acceptable", "borderline", "poor"):
            summary[f"sqi_{cls}_frac"] = float(np.mean(df_hrv["SQI_Class"] == cls))

    # Preserve scalar metadata from first row
    first = df_hrv.iloc[0]
    for col in df_hrv.columns:
        if col in _ARRAY_COLS or col in summary:
            continue
        val = first[col]
        if isinstance(val, (str, int, float, bool, np.integer, np.floating)):
            summary[col] = val

    return summary


def summarize_hrv_cohort(df_all_hrv):
    """
    Run summarize_hrv_record() for each participant and stack results.

    Parameters
    ----------
    df_all_hrv : pd.DataFrame
        Concatenated pipeline output. Must contain 'participant_id'.

    Returns
    -------
    pd.DataFrame or None
    """
    if df_all_hrv is None or len(df_all_hrv) == 0:
        return None

    if "participant_id" not in df_all_hrv.columns:
        raise ValueError("'participant_id' column is required.")

    summaries = []

    for pid, df_p in df_all_hrv.groupby("participant_id"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            s = summarize_hrv_record(df_p)
        if s is not None:
            s.setdefault("participant_id", pid)
            summaries.append(s)

    if not summaries:
        return None

    df_out = pd.DataFrame(summaries)
    cols   = ["participant_id"] + [c for c in df_out.columns if c != "participant_id"]
    return df_out[cols].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dominant_failure(df_hrv):
    """Most common non-good quality label, or 'none' if all good."""
    bad = df_hrv[~df_hrv["Quality"].isin(_HRV_QUALITY_LABELS)]["Quality"]
    if len(bad) == 0:
        return "none"
    return bad.value_counts().idxmax()


def _safe_mean(series):
    """Mean that returns np.nan rather than raising on all-NaN input."""
    vals = series.dropna()
    return float(vals.mean()) if len(vals) > 0 else np.nan


def _safe_median(series):
    """Median that returns np.nan on all-NaN input."""
    vals = series.dropna()
    return float(vals.median()) if len(vals) > 0 else np.nan