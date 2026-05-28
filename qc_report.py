"""
qc_report.py
------------
Generates a human-readable QC report from pipeline output.

Called automatically by pipeline.py at the end of every run.
Saved as <output_dir>/qc_report.txt

Collaborators can send this file instead of the full Excel output
when asking for help with their results.
"""

import os
import warnings
from datetime import datetime

import numpy as np
import pandas as pd


# ===========================================================================
# MAIN ENTRY POINT  (called by pipeline.py)
# ===========================================================================

def write_qc_report(df_all, config, output_dir):
    """
    Write a plain-text QC report summarising pipeline output.

    Parameters
    ----------
    df_all : pd.DataFrame
        Full pipeline output (all participants, all windows).
    config : dict
        Pipeline config dict.
    output_dir : str
        Directory to write qc_report.txt into.
    """
    lines = _build_report(df_all, config)
    report = "\n".join(lines)

    path = os.path.join(output_dir, "qc_report.txt")
    with open(path, "w") as f:
        f.write(report)

    print(f"[pipeline] Saved: qc_report.txt")
    return path


# ===========================================================================
# REPORT BUILDER
# ===========================================================================

def _build_report(df_all, config):
    lines = []
    now   = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines += [
        "=" * 64,
        "  nm_hrv — Pipeline QC report",
        f"  Generated: {now}",
        "=" * 64,
    ]

    # ---- Config snapshot --------------------------------------------------
    lines += [
        "",
        "CONFIG",
        f"  data_format      : {config.get('data_format','?')}",
        f"  window_sec       : {config.get('window_sec','?')}",
        f"  sqi_mode         : {config.get('sqi_mode','flag')}",
        f"  max_artifact_frac: {config.get('max_artifact_fraction','?')}",
        f"  template_corr_min: {config.get('template_corr_min','?')}",
        f"  sqi_qrs_min      : {config.get('sqi_qrs_min','?')}",
        f"  min_rr_for_hrv   : {config.get('min_rr_for_hrv','?')}",
        f"  ecg_peaks_method : {config.get('ecg_peaks_method','?')}",
    ]

    if not len(df_all):
        lines += ["", "No output rows — pipeline produced no results.", ""]
        return lines

    total      = len(df_all)
    n_parts    = df_all["participant_id"].nunique() if "participant_id" in df_all.columns else "?"
    hrv_mask   = df_all["Quality"].isin(["good", "good_low_sqi"])
    n_hrv      = int(hrv_mask.sum())

    # ---- Overall summary --------------------------------------------------
    lines += [
        "",
        "OVERALL",
        f"  Participants     : {n_parts}",
        f"  Total windows    : {total}",
        f"  Windows with HRV : {n_hrv}  ({n_hrv/total*100:.1f}%)",
    ]

    # ---- Quality breakdown ------------------------------------------------
    lines += ["", "QUALITY BREAKDOWN"]
    all_labels = [
        "good", "good_low_sqi", "poor_sqi",
        "too_many_artifacts", "peak_detection_failed", "hrv_failed",
    ]
    for lbl in all_labels:
        n   = int((df_all["Quality"] == lbl).sum())
        pct = n / total * 100
        lines.append(f"  {lbl:<28} {n:>6}  ({pct:5.1f}%)")

    # ---- Failure reason breakdown -----------------------------------------
    if "failure_reason" in df_all.columns:
        non_hrv = df_all[~hrv_mask]
        if len(non_hrv) > 0:
            lines += ["", "FAILURE REASONS (non-HRV windows)"]
            vc = non_hrv["failure_reason"].value_counts()
            for reason, cnt in vc.items():
                pct = cnt / len(non_hrv) * 100
                lines.append(f"  {reason:<28} {cnt:>6}  ({pct:5.1f}% of non-HRV)")

    # ---- SQI distribution -------------------------------------------------
    if "SQI_Class" in df_all.columns:
        lines += ["", "SQI CLASS DISTRIBUTION (all windows)"]
        for cls in ("excellent", "acceptable", "borderline", "poor"):
            n   = int((df_all["SQI_Class"] == cls).sum())
            pct = n / total * 100
            lines.append(f"  {cls:<28} {n:>6}  ({pct:5.1f}%)")

    # ---- HRV summary (windows with HRV only) ------------------------------
    df_hrv = df_all[hrv_mask]
    if len(df_hrv) > 0:
        lines += ["", "HRV METRICS (mean ± SD over windows with HRV computed)"]
        hrv_cols = [
            ("MeanHR_bpm",  "bpm"),
            ("RMSSD_ms",    "ms"),
            ("lnRMSSD",     ""),
            ("SDNN_ms",     "ms"),
            ("LF",          "ms²"),
            ("HF",          "ms²"),
            ("LF_HF",       ""),
            ("SD1",         "ms"),
            ("SD2",         "ms"),
            ("HRV_Confidence", ""),
        ]
        for col, unit in hrv_cols:
            if col in df_hrv.columns:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    vals = df_hrv[col].dropna()
                if len(vals) > 0:
                    suffix = f" {unit}" if unit else ""
                    lines.append(
                        f"  {col:<22} {vals.mean():>8.3f} ± {vals.std():>7.3f}{suffix}"
                    )

    # ---- Artifact fraction ------------------------------------------------
    if "ArtifactFraction" in df_hrv.columns and len(df_hrv) > 0:
        af = df_hrv["ArtifactFraction"].dropna()
        if len(af) > 0:
            lines += [
                "",
                "ARTIFACT CORRECTION (HRV windows)",
                f"  median ArtifactFraction : {af.median():.3f}",
                f"  mean   ArtifactFraction : {af.mean():.3f}",
                f"  p90    ArtifactFraction : {np.percentile(af, 90):.3f}",
            ]

    # ---- Per-participant summary ------------------------------------------
    lines += ["", "PER-PARTICIPANT SUMMARY"]
    lines.append(f"  {'ID':<20} {'windows':>7} {'HRV':>6} {'hrv%':>6} {'meanRMSSD':>10} {'confidence':>11}")
    lines.append(f"  {'-'*20} {'-'*7} {'-'*6} {'-'*6} {'-'*10} {'-'*11}")

    for pid, df_p in df_all.groupby("participant_id"):
        n_w     = len(df_p)
        hrv_p   = df_p["Quality"].isin(["good", "good_low_sqi"])
        n_h     = int(hrv_p.sum())
        pct_h   = n_h / n_w * 100 if n_w > 0 else 0

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rmssd = df_p.loc[hrv_p, "RMSSD_ms"].mean() if "RMSSD_ms" in df_p.columns else np.nan
            conf  = df_p.loc[hrv_p, "HRV_Confidence"].mean() if "HRV_Confidence" in df_p.columns else np.nan

        rmssd_s = f"{rmssd:>10.1f}" if not np.isnan(rmssd) else f"{'—':>10}"
        conf_s  = f"{conf:>11.3f}" if not np.isnan(conf) else f"{'—':>11}"

        # Flag participants with very low HRV yield
        flag = "  ⚠" if pct_h < 20 else "   "
        lines.append(f"{flag} {str(pid):<20} {n_w:>7} {n_h:>6} {pct_h:>5.0f}% {rmssd_s} {conf_s}")

    lines += [
        "",
        "  ⚠ = participant has < 20% windows with HRV (check data quality)",
    ]

    # ---- Actionable notes -------------------------------------------------
    notes = _generate_notes(df_all, df_hrv, config, total, n_hrv)
    if notes:
        lines += ["", "NOTES & RECOMMENDATIONS"]
        for note in notes:
            lines += [f"  {note}"]

    lines += [
        "",
        "=" * 64,
        "  Share this file (not the raw Excel) when asking for help.",
        "=" * 64,
        "",
    ]

    return lines


# ===========================================================================
# AUTOMATED NOTES
# ===========================================================================

def _generate_notes(df_all, df_hrv, config, total, n_hrv):
    """Generate actionable notes based on observed patterns."""
    notes = []
    hrv_pct = n_hrv / total * 100 if total > 0 else 0

    # Low overall HRV yield
    if hrv_pct < 30:
        notes.append(
            f"⚠ Low HRV yield ({hrv_pct:.0f}%). Run diagnose.py to identify "
            "the specific threshold causing rejections and get config recommendations."
        )

    # too_many_artifacts dominates
    n_art = int((df_all["Quality"] == "too_many_artifacts").sum())
    if n_art / total > 0.30:
        cfg_art = config.get("max_artifact_fraction", 0.05)
        notes.append(
            f"⚠ {n_art/total*100:.0f}% of windows rejected for too_many_artifacts. "
            f"Current max_artifact_fraction={cfg_art:.2f}. "
            f"For dry/wearable ECG consider raising to 0.15–0.20."
        )

    # low_template_corr dominates failures
    if "failure_reason" in df_all.columns:
        fr = df_all["failure_reason"].value_counts()
        if fr.get("low_template_corr", 0) / total > 0.20:
            cfg_tc = config.get("template_corr_min", 0.80)
            notes.append(
                f"⚠ Many windows failing template_corr check (threshold={cfg_tc:.2f}). "
                f"Dry/wearable ECG typically needs 0.50–0.65. "
                f"Consider lowering template_corr_min."
            )

    # HRV confidence is low
    if len(df_hrv) > 0 and "HRV_Confidence" in df_hrv.columns:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            med_conf = df_hrv["HRV_Confidence"].median()
        if not np.isnan(med_conf) and med_conf < 0.65:
            notes.append(
                f"⚠ Median HRV_Confidence = {med_conf:.2f} (< 0.65). "
                "Signal quality is low. Use HRV_Confidence as a covariate "
                "in your normative model rather than a hard exclusion threshold."
            )

    # Participants with zero HRV
    zero_hrv = []
    for pid, df_p in df_all.groupby("participant_id"):
        if not df_p["Quality"].isin(["good", "good_low_sqi"]).any():
            zero_hrv.append(str(pid))
    if zero_hrv:
        notes.append(
            f"⚠ {len(zero_hrv)} participant(s) have zero HRV windows: "
            f"{', '.join(zero_hrv[:5])}{'...' if len(zero_hrv) > 5 else ''}. "
            "Run diagnose.py on their files specifically."
        )

    if not notes:
        notes.append("✓ No major issues detected. Results look ready for harmonization.")

    return notes
