"""
pipeline.py
-----------
End-to-end HRV processing pipeline.

SQI mode (controlled via config key  sqi_mode):
  "flag"  (default) - SQI is computed and stored but never blocks HRV
                       extraction.  Every window that has detectable peaks
                       and acceptable artifact burden gets HRV metrics.
                       Quality label reflects SQI: "good" vs "good_low_sqi".
                       This is the correct mode for normative modelling across
                       heterogeneous cohorts (dry electrodes, clinical devices).
  "gate"             - classic hard-stop: windows failing SQI thresholds are
                       labelled "poor_sqi" and HRV is not computed.  Use only
                       if you need strict QC for a single clean-signal cohort.

Outputs
-------
  all_hrv_results.xlsx        - one row per (participant x window)
  participant_summaries.xlsx  - one row per participant
  rr_exports/<id>_seg_N.csv   - raw / corrected RR per window with HRV
  config_used.json            - frozen config snapshot
"""

import json
import os
import glob
import warnings

import numpy as np
import pandas as pd

from pathlib import Path

from .loaders       import load_ecg_record
from .preprocessing import preprocess_ecg
from .sqi           import compute_sqi, sqi_accept
from .rr            import detect_rr
from .hrv           import compute_hrv
from .summaries     import summarize_hrv_cohort
from .utils         import get_age_group
from .qc_report     import write_qc_report


# Columns that hold raw arrays - excluded from Excel / summaries
_ARRAY_COLS = {"ecg", "rr_ms_raw", "rr_ms_nn", "rr_flags"}

# HRV columns guaranteed to exist in every row with HRV computed (NaN-filled otherwise)
_EXPECTED_HRV_COLS = [
    "ArtifactFraction", "CorrectedBeatPct",
    "Beats_Total", "Beats_Corrected",
    "MeanHR_bpm", "RMSSD_ms", "SDNN_ms",
    "lnRMSSD", "lnSDNN",
    "RMSSD_c", "SDNN_c",
    "LF", "HF", "LF_HF", "VLF", "TP", "LF_nu", "HF_nu",
    "SD1", "SD2", "SD1_c",
    "HRV_Confidence",
]


# ===========================================================================
# CONFIDENCE SCORE
# ===========================================================================

def compute_hrv_confidence(sqi, artifact_fraction):
    """
    Composite HRV confidence score in [0, 1].

    Weights
    -------
      40 % SQI score (0-6 points, normalised)
      40 % (1 - artifact_fraction)
      20 % template correlation

    >= 0.80 - publication-quality data.
    """
    sqi_norm  = np.clip(sqi.get("SQI_Score", 0) / 6.0, 0, 1)
    art_score = np.clip(1.0 - artifact_fraction, 0, 1)
    tc        = sqi.get("template_corr", np.nan)
    tc_score  = np.clip(tc, 0, 1) if not np.isnan(tc) else 0.5

    return float(0.40 * sqi_norm + 0.40 * art_score + 0.20 * tc_score)


# ===========================================================================
# SINGLE-RECORD PROCESSOR
# ===========================================================================

def process_record(row, config):
    """
    Process one participant record into per-window HRV rows.

    With sqi_mode="flag" (default):
      - SQI metrics are always computed and stored.
      - HRV extraction proceeds regardless of SQI result.
      - Quality = "good"          if SQI passes and HRV succeeds
      - Quality = "good_low_sqi"  if SQI fails  but HRV succeeds
      - Quality = "too_many_artifacts" / "peak_detection_failed" /
                  "hrv_failed" if the signal cannot produce HRV at all.

    With sqi_mode="gate":
      - Original behaviour: windows failing SQI are labelled "poor_sqi"
        and HRV is not computed.
    """
    fs         = float(row["frequency"])
    ecg        = np.asarray(row["ecg"], dtype=float)
    window_sec = config["window_sec"]
    sqi_mode   = config.get("sqi_mode", "flag")   # "flag" | "gate"

    # ---- Duration guard ----------------------------------------------------
    duration_sec = len(ecg) / fs
    if duration_sec < window_sec:
        print(
            f"  [skip] {row['participant_id']}: "
            f"recording too short ({duration_sec:.1f}s < {window_sec}s required)"
        )
        return pd.DataFrame()

    # ---- ECG preprocessing -------------------------------------------------
    try:
        ecg_pp = preprocess_ecg(ecg, fs, cfg=config)
    except Exception as exc:
        print(f"  [warn] preprocessing failed for {row['participant_id']}: {exc}")
        return pd.DataFrame()

    samples_per_window     = int(fs * window_sec)
    age_col                = config.get("age_column", "age")
    age                    = row[age_col] if age_col in row.index else np.nan
    keep_first_window_only = config.get("keep_first_window_only", False)

    outputs = []

    for start in range(0, len(ecg_pp), samples_per_window):
        segment     = ecg_pp[start: start + samples_per_window]
        segment_idx = start // samples_per_window

        if keep_first_window_only and segment_idx > 0:
            break

        # Require ≥ 80 % window coverage
        if len(segment) < samples_per_window * 0.8:
            continue

        pid = row["participant_id"]

        base_meta = {
            "participant_id":   pid,
            "segment_idx":      segment_idx,
            "window_start_sec": float(start / fs),
            "window_sec":       float(window_sec),
            "AgeGroup":         get_age_group(age),
        }

        # ---- SQI (always computed) -----------------------------------------
        sqi          = compute_sqi(segment, fs)
        sqi_passed   = sqi_accept(sqi, config)
        failure_reason = _sqi_failure_reason(sqi, config) if not sqi_passed else "none"

        # ---- GATE mode: hard stop on SQI failure ---------------------------
        if sqi_mode == "gate" and not sqi_passed:
            out = {
                **base_meta,
                "Quality":        "poor_sqi",
                "failure_reason": failure_reason,
                **_prefix_sqi(sqi),
            }
            out = _attach_metadata(out, row, config)
            outputs.append(out)
            continue

        # ---- RR detection & artifact correction ----------------------------
        rr = detect_rr(segment, fs, config, age=age)

        if rr is None:
            out = {
                **base_meta,
                "Quality":        "peak_detection_failed",
                "failure_reason": "peak_detection_failed",
                "SQI_passed":     sqi_passed,
                **_prefix_sqi(sqi),
            }
            out = _attach_metadata(out, row, config)
            outputs.append(out)
            continue

        if rr["rr_ms_nn"] is None:
            out = {
                **base_meta,
                "Quality":           "too_many_artifacts",
                "failure_reason":    "artifact_fraction_exceeded",
                "SQI_passed":        sqi_passed,
                "ArtifactFraction":  rr["artifact_fraction"],
                "Beats_Total":       rr["n_beats"],
                "Beats_Corrected":   rr["n_artifacts"],
                **_prefix_sqi(sqi),
            }
            out = _attach_metadata(out, row, config)
            outputs.append(out)
            continue

        # ---- HRV extraction -----------------------------------------------
        hrv = compute_hrv(
            rr["rr_ms_nn"],
            min_nn     = config.get("min_rr_for_hrv", 150),
            lambda_val = config.get("smoothness_prior_lambda", 300),
            nperseg    = config.get("psd_nperseg", 256),
            noverlap   = config.get("psd_noverlap", 128),
            psd_window = config.get("psd_window", "hann"),
        )

        if hrv is None:
            out = {
                **base_meta,
                "Quality":        "hrv_failed",
                "failure_reason": "insufficient_nn_intervals",
                "SQI_passed":     sqi_passed,
                **_prefix_sqi(sqi),
            }
            out = _attach_metadata(out, row, config)
            outputs.append(out)
            continue

        # ---- Quality label -------------------------------------------------
        # In flag mode: HRV computed for all, quality reflects SQI result.
        # "good"         → SQI passed + HRV extracted
        # "good_low_sqi" → SQI failed + HRV extracted (use with caution;
        #                   HRV_Confidence score will be lower)
        quality        = "good" if sqi_passed else "good_low_sqi"
        confidence     = compute_hrv_confidence(sqi, rr["artifact_fraction"])
        corrected_pct  = rr["n_artifacts"] / max(rr["n_beats"], 1) * 100.0

        out = {
            **base_meta,
            "Quality":              quality,
            "SQI_passed":           sqi_passed,
            "failure_reason":       failure_reason,
            **_prefix_sqi(sqi),
            **hrv,
            "ArtifactFraction":     rr["artifact_fraction"],
            "CorrectedBeatPct":     corrected_pct,
            "Beats_Total":          rr["n_beats"],
            "Beats_Corrected":      rr["n_artifacts"],
            "PeakDetectionMethod":  config.get("ecg_peaks_method", "pantompkins1985"),
            "DetectedPeaks":        int(len(rr["rpeaks"])),
            "HRV_Confidence":       confidence,
            # Raw arrays stripped before Excel write, kept for RR export
            "rr_ms_raw":            rr["rr_ms_raw"],
            "rr_ms_nn":             rr["rr_ms_nn"],
            "rr_flags":             rr["rr_flags"],
        }
        out = _attach_metadata(out, row, config)
        outputs.append(out)

    return pd.DataFrame(outputs)


# ===========================================================================
# FULL COHORT PIPELINE
# ===========================================================================

def run_pipeline(config):
    """
    Run the complete HRV pipeline for a cohort.
    """
    data_format  = config["data_format"]
    data_folder  = config["data_folder"]
    metadata_csv = config.get("metadata_csv", "")
    output_dir   = config["output_dir"]
    sqi_mode     = config.get("sqi_mode", "flag")

    os.makedirs(output_dir, exist_ok=True)

    # Freeze config snapshot
    with open(os.path.join(output_dir, "config_used.json"), "w") as f:
        json.dump(config, f, indent=4, default=str)

    # Load optional metadata
    if metadata_csv and os.path.exists(metadata_csv):
        df_metadata = pd.read_csv(metadata_csv)
        df_metadata["participant_id"] = df_metadata["participant_id"].astype(str)
        print(f"[pipeline] Loaded metadata: {len(df_metadata)} participants")
    else:
        df_metadata = None
        if metadata_csv:
            print(f"[pipeline] Metadata file not found, proceeding without: {metadata_csv}")

    # Discover files
    if data_format == "edf":
        files = sorted(glob.glob(os.path.join(data_folder, "*.edf")))
    elif data_format == "wfdb":
        hea_files = sorted(glob.glob(os.path.join(data_folder, "*.hea")))
        files     = [f.replace(".hea", "") for f in hea_files]
    else:
        raise ValueError(f"Unsupported data_format: '{data_format}'")

    if not files:
        print(f"[pipeline] No {data_format.upper()} files found in {data_folder}")
        return

    print(f"[pipeline] Found {len(files)} file(s) to process.  SQI mode: {sqi_mode}")

    all_results = []

    for file in files:
        print(f"\n[pipeline] Processing: {Path(file).name}")
        try:
            row       = load_ecg_record(file, data_format, df_metadata)
            df_result = process_record(row, config)

            if df_result is not None and len(df_result) > 0:
                all_results.append(df_result)
                n_good     = int((df_result["Quality"] == "good").sum())
                n_low_sqi  = int((df_result["Quality"] == "good_low_sqi").sum())
                n_hrv      = n_good + n_low_sqi
                print(f"  → {len(df_result)} windows | "
                      f"{n_good} good | {n_low_sqi} good_low_sqi | "
                      f"{n_hrv} total with HRV")

        except Exception as exc:
            print(f"  [ERROR] {Path(file).name}: {exc}")

    if not all_results:
        print("\n[pipeline] No valid outputs produced. Check data and config.")
        return

    df_all = pd.concat(all_results, ignore_index=True)

    # Ensure expected HRV columns exist (NaN for non-HRV rows)
    for col in _EXPECTED_HRV_COLS:
        if col not in df_all.columns:
            df_all[col] = np.nan

    # Save per-window results (strip raw arrays)
    df_excel = df_all.drop(
        columns=[c for c in _ARRAY_COLS if c in df_all.columns],
        errors="ignore",
    )
    xlsx_path = os.path.join(output_dir, "all_hrv_results.xlsx")
    df_excel.to_excel(xlsx_path, index=False)
    print(f"\n[pipeline] Saved: all_hrv_results.xlsx ({len(df_excel)} rows)")

    # Save cohort summaries (suppress nanmean-of-empty-slice warnings)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        df_summary = summarize_hrv_cohort(df_all)

    if df_summary is not None:
        df_summary.to_excel(
            os.path.join(output_dir, "participant_summaries.xlsx"),
            index=False,
        )
        print(f"[pipeline] Saved: participant_summaries.xlsx "
              f"({len(df_summary)} participants)")

    # Export RR CSVs
    _export_rr_csvs(df_all, output_dir)

    # Write QC report
    write_qc_report(df_all, config, output_dir)

    # Print cohort summary
    _print_cohort_summary(df_all)

    print("\n[pipeline] DONE.")


# ===========================================================================
# RR CSV EXPORT
# ===========================================================================

def _export_rr_csvs(df_all, output_dir):
    """Write one CSV per window that has HRV computed."""
    rr_dir = os.path.join(output_dir, "rr_exports")
    os.makedirs(rr_dir, exist_ok=True)

    n_exported = 0

    for _, row in df_all.iterrows():
        if "rr_ms_raw" not in row.index:
            continue

        rr_raw = row["rr_ms_raw"]
        if rr_raw is None:
            continue
        if isinstance(rr_raw, float) and np.isnan(rr_raw):
            continue
        if not isinstance(rr_raw, (list, np.ndarray)):
            continue

        n        = len(rr_raw)
        rr_nn    = row.get("rr_ms_nn",  None)
        rr_flags = row.get("rr_flags",  None)

        rr_nn_vals   = list(rr_nn)    if isinstance(rr_nn,   (list, np.ndarray)) else [np.nan] * n
        rr_flag_vals = list(rr_flags) if isinstance(rr_flags,(list, np.ndarray)) else [np.nan] * n

        rr_nn_vals   = (rr_nn_vals   + [np.nan] * n)[:n]
        rr_flag_vals = (rr_flag_vals + [np.nan] * n)[:n]

        rr_df = pd.DataFrame({
            "rr_ms_raw": rr_raw,
            "rr_ms_nn":  rr_nn_vals,
            "rr_flags":  rr_flag_vals,
        })

        fname = (
            f"{row['participant_id']}"
            f"_segment_{row['segment_idx']}"
            ".csv"
        )
        rr_df.to_csv(os.path.join(rr_dir, fname), index=False)
        n_exported += 1

    print(f"[pipeline] Saved: {n_exported} RR export CSV(s) - rr_exports/")


# ===========================================================================
# HELPERS
# ===========================================================================

def _prefix_sqi(sqi):
    """Return SQI dict with scalar values only."""
    return {
        k: v for k, v in sqi.items()
        if isinstance(v, (int, float, bool, str, np.integer, np.floating))
        and k not in _ARRAY_COLS
    }


def _attach_metadata(out, row, config=None):
    """
    Copy metadata columns from the loader row into the output dict.

    If config contains 'metadata_columns_to_carry' (a list), only those
    columns are copied.  Empty list or absent key - copy all scalar columns.
    """
    carry = None
    if config is not None:
        carry = config.get("metadata_columns_to_carry", None)
        if carry is not None and len(carry) == 0:
            carry = None

    for col in row.index:
        if col in _ARRAY_COLS or col in out:
            continue
        if carry is not None and col not in carry:
            continue
        val = row[col]
        if isinstance(val, (str, int, float, bool, np.integer, np.floating)):
            out[col] = val
    return out


def _sqi_failure_reason(sqi, config):
    """Return the first failed SQI gate as a descriptive string."""
    if sqi["flatline_fraction"] > config["max_flatline_fraction"]:
        return "flatline"
    if sqi["clipping_fraction"] > config["max_clipping_fraction"]:
        return "clipping"
    if sqi["qrs_power_ratio"] < config["sqi_qrs_min"]:
        return "low_qrs_power"
    if not sqi["hr_plausible"]:
        return "implausible_hr"
    if sqi["n_rpeaks"] < config["min_rpeaks"]:
        return "too_few_peaks"
    tc = sqi.get("template_corr", np.nan)
    if np.isnan(tc) or tc < config["template_corr_min"]:
        return "low_template_corr"
    return "unknown_sqi"


def _print_cohort_summary(df_all):
    """Print quality breakdown to stdout."""
    total = len(df_all)
    if total == 0:
        return

    # Windows with HRV computed = good + good_low_sqi
    has_hrv = df_all["Quality"].isin(["good", "good_low_sqi"])
    n_hrv   = int(has_hrv.sum())

    print("\n--- Cohort Quality Summary ---------------------------------")
    all_labels = [
        "good", "good_low_sqi",
        "poor_sqi",
        "too_many_artifacts",
        "peak_detection_failed",
        "hrv_failed",
    ]
    for label in all_labels:
        n   = int((df_all["Quality"] == label).sum())
        pct = n / total * 100
        print(f"  {label:<28} {n:>6}  ({pct:5.1f}%)")

    print(f"  {'-'*46}")
    print(f"  {'TOTAL':<28} {total:>6}")
    print(f"  {'Windows with HRV computed':<28} {n_hrv:>6}  ({n_hrv/total*100:5.1f}%)")
    print("-------------------------------------------------------------------")

    if "HRV_Confidence" in df_all.columns:
        conf = df_all.loc[has_hrv, "HRV_Confidence"].dropna()
        if len(conf) > 0:
            print(f"  HRV Confidence (HRV windows): "
                  f"mean={conf.mean():.3f}  median={conf.median():.3f}")

    if "failure_reason" in df_all.columns:
        no_hrv = df_all[~has_hrv & (df_all["Quality"] != "poor_sqi")]
        if len(no_hrv) > 0:
            top = no_hrv["failure_reason"].value_counts().head(3)
            print(f"\n  Top failure reasons (non-SQI):")
            for reason, cnt in top.items():
                print(f"    {reason:<30} {cnt}")

    print()