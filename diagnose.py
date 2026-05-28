"""
diagnose.py
-----------
Pre-run diagnostic tool for nm_hrv.

Run this BEFORE run_pipeline.py on a new dataset.
It samples up to `n_sample` files, measures actual SQI values,
compares them against your config thresholds, and prints a plain-language
report with specific config recommendations.

Usage
-----
    python diagnose.py --config configs/edf_config.yaml
    python diagnose.py --config configs/edf_config.yaml --n_sample 5

The report is also saved as  <output_dir>/diagnostic_report.txt
so collaborators can send it to you directly.

No data leaves the machine - the report contains only aggregate statistics,
no raw ECG samples.
"""

import argparse
import glob
import os
import sys
import textwrap
from pathlib import Path

import numpy as np
import yaml

from nm_hrv.loaders      import load_ecg_record
from nm_hrv.preprocessing import preprocess_ecg
from nm_hrv.sqi          import compute_sqi, sqi_accept
from nm_hrv.rr           import detect_rr


# ===========================================================================
# HELPERS
# ===========================================================================

def _pct(v, total):
    return f"{v/total*100:.0f}%" if total > 0 else "n/a"


def _p(label, value, width=36):
    """Format a key-value line for the report."""
    return f"  {label:<{width}} {value}"


def _header(title):
    bar = "─" * 62
    return f"\n{bar}\n  {title}\n{bar}"


def _check(condition):
    return "✓" if condition else "✗"


# ===========================================================================
# SINGLE-FILE SAMPLER
# ===========================================================================

def sample_file(file, data_format, config, window_sec):
    """
    Load one file and compute SQI + RR stats on its first 3 windows.
    Returns a list of dicts (one per window sampled).
    """
    results = []

    try:
        row    = load_ecg_record(file, data_format, df_metadata=None)
        fs     = float(row["frequency"])
        ecg    = np.asarray(row["ecg"], dtype=float)
        dur    = len(ecg) / fs

        if dur < window_sec * 0.8:
            return [{"file": Path(file).name, "error": f"too short ({dur:.0f}s)"}]

        ecg_pp = preprocess_ecg(ecg, fs, cfg=config)
        spw    = int(fs * window_sec)

        for start in range(0, min(len(ecg_pp), spw * 3), spw):
            seg = ecg_pp[start: start + spw]
            if len(seg) < spw * 0.8:
                continue

            sqi     = compute_sqi(seg, fs)
            passed  = sqi_accept(sqi, config)

            rr_info = None
            art_frac = np.nan
            if sqi.get("n_rpeaks", 0) >= 3:
                try:
                    rr_info  = detect_rr(seg, fs, config)
                    if rr_info is not None:
                        art_frac = rr_info["artifact_fraction"]
                except Exception:
                    pass

            results.append({
                "file":            Path(file).name,
                "fs":              fs,
                "duration_sec":    dur,
                "sqi_passed":      passed,
                "qrs_power_ratio": sqi.get("qrs_power_ratio", np.nan),
                "template_corr":   sqi.get("template_corr",   np.nan),
                "snr_db":          sqi.get("snr_db",          np.nan),
                "flatline":        sqi.get("flatline_fraction",np.nan),
                "clipping":        sqi.get("clipping_fraction",np.nan),
                "hr_plausible":    sqi.get("hr_plausible",    False),
                "peak_density":    sqi.get("peak_density_bpm",np.nan),
                "n_rpeaks":        sqi.get("n_rpeaks",        0),
                "rr_cv":           sqi.get("rr_cv",           np.nan),
                "artifact_frac":   art_frac,
                "SQI_Class":       sqi.get("SQI_Class",       "unknown"),
                "SQI_Score":       sqi.get("SQI_Score",       0),
            })

    except Exception as exc:
        results.append({"file": Path(file).name, "error": str(exc)})

    return results


# ===========================================================================
# RECOMMENDATIONS ENGINE
# ===========================================================================

def make_recommendations(rows, config):
    """
    Compare observed SQI distributions against config thresholds.
    Returns a list of (severity, parameter, observation, recommendation) tuples.
    severity: "ERROR" | "WARN" | "OK"
    """
    good_rows = [r for r in rows if "error" not in r]
    if not good_rows:
        return [("ERROR", "data", "No files could be loaded", "Check data_folder path and data_format in config")]

    recs = []

    def med(key):
        vals = [r[key] for r in good_rows if not np.isnan(r[key])]
        return float(np.median(vals)) if vals else np.nan

    def pct_failing(key, threshold, direction="below"):
        vals = [r[key] for r in good_rows if not np.isnan(r[key])]
        if not vals:
            return np.nan
        if direction == "below":
            return np.mean([v < threshold for v in vals]) * 100
        else:
            return np.mean([v > threshold for v in vals]) * 100

    n_total   = len(good_rows)
    n_passed  = sum(r["sqi_passed"] for r in good_rows)
    pass_rate = n_passed / n_total * 100 if n_total > 0 else 0

    # ---- SQI pass rate overall --------------------------------------------
    if pass_rate < 20:
        recs.append(("ERROR", "overall SQI pass rate",
                     f"{pass_rate:.0f}% of sampled windows pass SQI gate",
                     "Multiple thresholds likely too strict for this device/format. See specifics below."))
    elif pass_rate < 60:
        recs.append(("WARN", "overall SQI pass rate",
                     f"{pass_rate:.0f}% of sampled windows pass SQI gate",
                     "Some thresholds may need loosening. See specifics below."))
    else:
        recs.append(("OK", "overall SQI pass rate",
                     f"{pass_rate:.0f}% of sampled windows pass SQI gate",
                     "Acceptable — no change needed"))

    # ---- template_corr ----------------------------------------------------
    med_tc    = med("template_corr")
    cfg_tc    = config.get("template_corr_min", 0.80)
    pct_tc    = pct_failing("template_corr", cfg_tc, "below")

    if not np.isnan(med_tc):
        if med_tc < cfg_tc:
            suggested = max(0.50, round(med_tc - 0.05, 2))
            recs.append(("ERROR", "template_corr_min",
                         f"median template_corr = {med_tc:.2f}, threshold = {cfg_tc:.2f} "
                         f"({pct_tc:.0f}% of windows fail this check)",
                         f"Lower template_corr_min to {suggested:.2f}  "
                         f"(typical for dry/wearable ECG: 0.50–0.65; gel/Holter: 0.80)"))
        elif pct_tc > 20:
            suggested = max(0.50, round(med_tc - 0.10, 2))
            recs.append(("WARN", "template_corr_min",
                         f"median template_corr = {med_tc:.2f} but {pct_tc:.0f}% of windows fail",
                         f"Consider lowering template_corr_min to {suggested:.2f}"))
        else:
            recs.append(("OK", "template_corr_min",
                         f"median template_corr = {med_tc:.2f} (threshold {cfg_tc:.2f})",
                         "No change needed"))

    # ---- qrs_power_ratio --------------------------------------------------
    med_qrs   = med("qrs_power_ratio")
    cfg_qrs   = config.get("sqi_qrs_min", 0.25)
    pct_qrs   = pct_failing("qrs_power_ratio", cfg_qrs, "below")

    if not np.isnan(med_qrs):
        if med_qrs < cfg_qrs:
            suggested = max(0.10, round(med_qrs - 0.05, 2))
            recs.append(("ERROR", "sqi_qrs_min",
                         f"median qrs_power_ratio = {med_qrs:.2f}, threshold = {cfg_qrs:.2f} "
                         f"({pct_qrs:.0f}% fail)",
                         f"Lower sqi_qrs_min to {suggested:.2f}"))
        elif pct_qrs > 20:
            recs.append(("WARN", "sqi_qrs_min",
                         f"median qrs_power_ratio = {med_qrs:.2f} but {pct_qrs:.0f}% of windows fail",
                         f"Consider lowering sqi_qrs_min to {max(0.10, round(cfg_qrs - 0.05, 2)):.2f}"))
        else:
            recs.append(("OK", "sqi_qrs_min",
                         f"median qrs_power_ratio = {med_qrs:.2f} (threshold {cfg_qrs:.2f})",
                         "No change needed"))

    # ---- artifact_fraction ------------------------------------------------
    med_art   = med("artifact_frac")
    cfg_art   = config.get("max_artifact_fraction", 0.05)
    pct_art   = pct_failing("artifact_frac", cfg_art, "above")

    if not np.isnan(med_art):
        if med_art > cfg_art:
            suggested = min(0.30, round(med_art + 0.05, 2))
            recs.append(("ERROR", "max_artifact_fraction",
                         f"median artifact_fraction = {med_art:.2f}, limit = {cfg_art:.2f} "
                         f"({pct_art:.0f}% of windows exceed limit)",
                         f"Raise max_artifact_fraction to {suggested:.2f}  "
                         f"(dry/wearable ECG: 0.15–0.20; Holter: 0.05–0.10)"))
        elif pct_art > 30:
            recs.append(("WARN", "max_artifact_fraction",
                         f"median artifact_fraction = {med_art:.2f} but {pct_art:.0f}% of windows exceed limit",
                         f"Consider raising max_artifact_fraction to {min(0.30, round(cfg_art + 0.05, 2)):.2f}"))
        else:
            recs.append(("OK", "max_artifact_fraction",
                         f"median artifact_fraction = {med_art:.2f} (limit {cfg_art:.2f})",
                         "No change needed"))

    # ---- HR plausibility --------------------------------------------------
    pct_implausible = np.mean([not r["hr_plausible"] for r in good_rows]) * 100
    med_hr = med("peak_density")

    if pct_implausible > 30:
        recs.append(("ERROR", "hr_plausible",
                     f"{pct_implausible:.0f}% of windows have implausible HR "
                     f"(median detected HR = {med_hr:.0f} bpm)",
                     "Peak detection may be failing. Check ecg_peaks_method. "
                     "If recording has high HR (children/exercise), confirm rr_min_ms is appropriate."))
    elif pct_implausible > 10:
        recs.append(("WARN", "hr_plausible",
                     f"{pct_implausible:.0f}% of windows have implausible HR",
                     "Check for noisy segments or wrong channel selection"))
    else:
        recs.append(("OK", "hr_plausible",
                     f"median HR = {med_hr:.0f} bpm, {pct_implausible:.0f}% implausible",
                     "No change needed"))

    # ---- flatline ---------------------------------------------------------
    med_flat  = med("flatline")
    cfg_flat  = config.get("max_flatline_fraction", 0.05)
    pct_flat  = pct_failing("flatline", cfg_flat, "above")

    if pct_flat > 20:
        recs.append(("WARN", "max_flatline_fraction",
                     f"{pct_flat:.0f}% of windows exceed flatline threshold (median = {med_flat:.3f})",
                     "Signal may have dropout segments. Check electrode contact quality. "
                     "If expected for this device, raise max_flatline_fraction slightly."))
    else:
        recs.append(("OK", "max_flatline_fraction",
                     f"median flatline_fraction = {med_flat:.3f} (limit {cfg_flat:.2f})",
                     "No change needed"))

    # ---- SNR --------------------------------------------------------------
    med_snr = med("snr_db")
    if not np.isnan(med_snr) and med_snr < 5:
        recs.append(("WARN", "snr_db (informational)",
                     f"median SNR = {med_snr:.1f} dB (low — typical clean ECG > 10 dB)",
                     "Low SNR is expected for dry/wearable devices. No config change needed "
                     "unless SNR is < 0 dB (signal dominated by noise)."))
    elif not np.isnan(med_snr):
        recs.append(("OK", "snr_db",
                     f"median SNR = {med_snr:.1f} dB",
                     "No change needed"))

    # ---- Sampling rate ----------------------------------------------------
    fs_vals = list({r["fs"] for r in good_rows if "fs" in r})
    if len(fs_vals) > 1:
        recs.append(("WARN", "sampling_rate",
                     f"Mixed sampling rates detected: {sorted(fs_vals)}",
                     "Pipeline handles this correctly, but note it in your methods."))
    else:
        recs.append(("OK", "sampling_rate",
                     f"Consistent sampling rate: {fs_vals[0] if fs_vals else 'unknown'} Hz",
                     "No change needed"))

    return recs


# ===========================================================================
# REPORT FORMATTER
# ===========================================================================

def format_report(rows, recs, config, files_sampled, files_total):
    """Render a plain-text diagnostic report."""
    good_rows = [r for r in rows if "error" not in r]
    error_rows = [r for r in rows if "error" in r]

    lines = []
    lines.append("=" * 64)
    lines.append("  nm_hrv — Pre-run diagnostic report")
    lines.append("=" * 64)
    lines.append(f"\n  Config:          {config.get('data_format','?').upper()}")
    lines.append(f"  Files found:     {files_total}")
    lines.append(f"  Files sampled:   {files_sampled}")
    lines.append(f"  Windows sampled: {len(rows)}")
    lines.append(f"  Windows OK:      {len(good_rows)}")
    if error_rows:
        lines.append(f"  Load errors:     {len(error_rows)}")
        for r in error_rows:
            lines.append(f"    ✗ {r['file']}: {r['error']}")

    # ---- SQI score distribution ------------------------------------------
    lines.append(_header("SQI score distribution (0 = poor → 6 = excellent)"))
    if good_rows:
        scores = [r["SQI_Score"] for r in good_rows]
        for s in range(7):
            cnt = scores.count(s)
            bar = "█" * cnt
            lines.append(f"  {s}  {bar:<30} {cnt:>3}  ({_pct(cnt, len(scores))})")

        classes = [r["SQI_Class"] for r in good_rows]
        lines.append(f"\n  excellent : {_pct(classes.count('excellent'), len(classes))}")
        lines.append(f"  acceptable: {_pct(classes.count('acceptable'), len(classes))}")
        lines.append(f"  borderline: {_pct(classes.count('borderline'), len(classes))}")
        lines.append(f"  poor      : {_pct(classes.count('poor'),       len(classes))}")

    # ---- Observed SQI values vs thresholds --------------------------------
    lines.append(_header("Observed SQI values vs current config thresholds"))

    def stat_line(label, key, threshold, direction, cfg_key):
        vals = [r[key] for r in good_rows if not np.isnan(r.get(key, np.nan))]
        if not vals:
            return f"  {label:<28} no data"
        med = np.median(vals)
        p10 = np.percentile(vals, 10)
        p90 = np.percentile(vals, 90)
        if direction == "above":
            failing = np.mean([v > threshold for v in vals]) * 100
        else:
            failing = np.mean([v < threshold for v in vals]) * 100
        ok = "✓" if failing < 20 else "✗"
        return (f"  {ok} {label:<26} "
                f"median={med:.2f}  p10={p10:.2f}  p90={p90:.2f}  "
                f"threshold={threshold:.2f}  failing={failing:.0f}%")

    lines.append(stat_line("template_corr",   "template_corr",   config.get("template_corr_min",0.80),    "below", "template_corr_min"))
    lines.append(stat_line("qrs_power_ratio",  "qrs_power_ratio", config.get("sqi_qrs_min",0.25),         "below", "sqi_qrs_min"))
    lines.append(stat_line("artifact_fraction","artifact_frac",   config.get("max_artifact_fraction",0.05),"above", "max_artifact_fraction"))
    lines.append(stat_line("flatline_fraction","flatline",        config.get("max_flatline_fraction",0.05),"above", "max_flatline_fraction"))

    snr_vals = [r["snr_db"] for r in good_rows if not np.isnan(r.get("snr_db", np.nan))]
    if snr_vals:
        lines.append(f"  {'snr_db (info only)':<28} "
                     f"median={np.median(snr_vals):.1f} dB  "
                     f"p10={np.percentile(snr_vals,10):.1f}  "
                     f"p90={np.percentile(snr_vals,90):.1f}")

    hr_vals = [r["peak_density"] for r in good_rows if not np.isnan(r.get("peak_density", np.nan))]
    if hr_vals:
        lines.append(f"  {'peak HR (info only)':<28} "
                     f"median={np.median(hr_vals):.0f} bpm  "
                     f"p10={np.percentile(hr_vals,10):.0f}  "
                     f"p90={np.percentile(hr_vals,90):.0f}")

    # ---- Recommendations --------------------------------------------------
    lines.append(_header("Recommendations"))

    errors = [r for r in recs if r[0] == "ERROR"]
    warns  = [r for r in recs if r[0] == "WARN"]
    oks    = [r for r in recs if r[0] == "OK"]

    if errors:
        lines.append("\n  MUST FIX (pipeline will produce very few HRV outputs):\n")
        for _, param, obs, fix in errors:
            lines.append(f"  ✗  {param}")
            lines.append(f"     Observed:    {obs}")
            lines.append(f"     Recommended: {fix}\n")

    if warns:
        lines.append("  CONSIDER (may improve HRV yield):\n")
        for _, param, obs, fix in warns:
            lines.append(f"  ⚠  {param}")
            lines.append(f"     Observed:    {obs}")
            lines.append(f"     Recommended: {fix}\n")

    if oks and not errors and not warns:
        lines.append("\n  ✓  All parameters look appropriate for this dataset.")
        lines.append("     Proceed with run_pipeline.py\n")
    elif oks:
        lines.append("  OK (no change needed):")
        for _, param, obs, _ in oks:
            lines.append(f"  ✓  {param}: {obs}")

    # ---- Suggested config block -------------------------------------------
    error_params = {r[1]: r[3] for r in errors}
    warn_params  = {r[1]: r[3] for r in warns}

    if error_params or warn_params:
        lines.append(_header("Suggested config changes  (copy into your YAML)"))
        lines.append("")

        param_map = {
            "template_corr_min":      "template_corr_min",
            "sqi_qrs_min":            "sqi_qrs_min",
            "max_artifact_fraction":  "max_artifact_fraction",
            "max_flatline_fraction":  "max_flatline_fraction",
        }

        for label, fix_text in {**error_params, **warn_params}.items():
            cfg_key = param_map.get(label)
            if cfg_key:
                # Extract suggested numeric value from fix text
                import re
                match = re.search(r"(\d+\.\d+)", fix_text)
                if match:
                    lines.append(f"  {cfg_key}: {match.group(1)}")

        lines.append("")

    lines.append("=" * 64)
    lines.append("  Send this file to your collaborator if the pipeline")
    lines.append("  is not producing enough HRV outputs.")
    lines.append("=" * 64)

    return "\n".join(lines)


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="nm_hrv pre-run diagnostic — checks SQI values against config thresholds",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Run this before run_pipeline.py on any new dataset.
        The report is saved to <output_dir>/diagnostic_report.txt

        Examples:
          python diagnose.py --config configs/edf_config.yaml
          python diagnose.py --config configs/wfdb_config.yaml --n_sample 10
        """),
    )
    parser.add_argument("--config",   required=True, help="Path to YAML config file")
    parser.add_argument("--n_sample", type=int, default=5,
                        help="Number of files to sample (default: 5)")
    args = parser.parse_args()

    # Load config
    try:
        with open(args.config) as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"[diagnose] ERROR: config not found: {args.config}")
        sys.exit(1)

    data_format = config["data_format"]
    data_folder = config["data_folder"]
    output_dir  = config.get("output_dir", ".")
    window_sec  = config.get("window_sec", 300)
    n_sample    = args.n_sample

    # Discover files
    if data_format == "edf":
        files = sorted(glob.glob(os.path.join(data_folder, "*.edf")))
    elif data_format == "wfdb":
        hea = sorted(glob.glob(os.path.join(data_folder, "*.hea")))
        files = [f.replace(".hea", "") for f in hea]
    else:
        print(f"[diagnose] ERROR: unsupported data_format: {data_format}")
        sys.exit(1)

    if not files:
        print(f"[diagnose] ERROR: no {data_format.upper()} files found in: {data_folder}")
        sys.exit(1)

    files_total   = len(files)
    files_sampled = min(n_sample, files_total)
    sample_files  = files[:files_sampled]

    print(f"\n[diagnose] Sampling {files_sampled} of {files_total} "
          f"{data_format.upper()} file(s) from: {data_folder}")
    print(f"[diagnose] Window: {window_sec}s   Config: {args.config}\n")

    # Sample each file
    all_rows = []
    for file in sample_files:
        print(f"  Sampling: {Path(file).name} ...", end=" ", flush=True)
        rows = sample_file(file, data_format, config, window_sec)
        n_ok = sum(1 for r in rows if "error" not in r)
        print(f"{n_ok} windows")
        all_rows.extend(rows)

    # Generate recommendations
    recs   = make_recommendations(all_rows, config)
    report = format_report(all_rows, recs, config, files_sampled, files_total)

    # Print to console
    print("\n" + report)

    # Save report
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "diagnostic_report.txt")
    with open(report_path, "w") as f:
        f.write(report)

    print(f"\n[diagnose] Report saved: {report_path}")

    # Exit code: 1 if any ERRORs, so CI/scripts can detect problems
    has_errors = any(r[0] == "ERROR" for r in recs)
    sys.exit(1 if has_errors else 0)


if __name__ == "__main__":
    main()
