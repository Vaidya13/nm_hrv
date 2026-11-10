# nm_hrv

`nm_hrv` is a Python package for large-scale ECG heart rate variability (HRV)
analysis across diverse cohorts (adults, adolescents, children).

It implements the Task Force (1996) / Kubios standard frequency-domain
pipeline:

1. ECG preprocessing (bandpass + notch + robust normalisation)
2. R-peak detection (Pan-Tompkins via neurokit2)
3. Artifact classification with adaptive Lipponen & Tarvainen (2019) thresholds
4. Cubic spline interpolation of artifact beats
5. 4 Hz resampling onto a regular time grid
6. Smoothness-priors detrending (Tarvainen et al. 2002)
7. Welch PSD — Hann window, 256 samples (64s at 4HZ), 50% overlap
8. LF / HF / VLF power extraction (Task Force bands)
9. Nonlinear HRV (SD1, SD2, SD1/SD2)
10. Hierarchical SQI classification + HRV confidence score
11. Cohort-level summaries with usable-fraction QC


## Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-org/nm_hrv.git
cd nm_hrv

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install the package (editable mode recommended for development)
pip install -e .
```

---

## Quick start

```bash
# EDF cohort
python run_pipeline.py --config configs/edf_config.yaml

# WFDB cohort
python run_pipeline.py --config configs/wfdb_config.yaml
```

Edit the relevant config YAML to point at your data folder and metadata CSV
before running.

---

## Configuration

All settings live in a single YAML file.  Key parameters:

| Key | Default | Description |
|-----|---------|-------------|
| `data_format` | — | `"edf"` or `"wfdb"` |
| `window_sec` | `300` | Analysis window length (5 min = Task Force standard) |
| `segments_csv` | `""` | Analyse only the listed segments (see below); empty = whole recording |
| `segment_windowing` | `"tile"` | `"tile"`: `window_sec` windows inside each segment; `"whole"`: one window per segment |
| `min_window_sec` | `60` | Segments / partial windows shorter than this are not analysed |
| `max_artifact_fraction` | `0.05` | Window rejected if > 5 % beats are artefacts |
| `enable_age_adaptive_rules` | `true` | Adaptive RR bounds for children / adolescents |
| `smoothness_prior_lambda` | `300` | Kubios detrending strength |
| `ecg_peaks_method` | `"pantompkins1985"` | neurokit2 peak detector |
| `notch_freq` | `50.0` | `60.0` for US recordings |

---

## Analysing part of a recording (e.g. resting state only)

This is optional. Without `segments_csv` the whole recording is analysed. If a recording contains rest and task in one file, give the pipeline a CSV
of the parts to analyse and set `segments_csv` in the config:

```
participant_id,start_sec,stop_sec,label
P001,120,480,rest
P001,1500,1700,rest_2
P002,00:03:00,00:08:30,rest
```

- `participant_id` must equal the file name without extension.
- Times are measured *from the start of the recording* (not clock time), in seconds
  or as `MM:SS` / `HH:MM:SS`. One participant can have several rows. Other columns
  (e.g. `label`) are copied into every output row. Files that are not listed are skipped.
- Set `min_rr_for_hrv: 30` in the config; the default (150) rejects short windows.

| Segment length | Result |
|---|---|
| shorter than `min_window_sec` (60 s) | not analysed; listed in `skipped_segments.csv` with the reason |
| between `min_window_sec` and `window_sec` | one shorter window |
| longer than `window_sec` | consecutive `window_sec` windows; a final partial window is kept if >= `min_window_sec` (or use `segment_windowing: whole` for one window per segment) |

Output rows also carry `segment_id` (order of the segment within that participant's
rows), `window_in_segment`, `segment_start_sec`, `segment_stop_sec` and
`segment_analysed_sec`, and `window_sec` holds the actual length analysed. The whole
recording is filtered first and then cut, so no filter edge effects appear at
segment boundaries.

**Windows shorter than 5 minutes.** HRV metrics depend on analysis duration, and the
pipeline does not exclude anything on duration: all metrics are computed. Decide
per metric which rows to use, using `window_sec` and `NN_Count`.

---

## Outputs

| File | Description |
|------|-------------|
| `all_hrv_results.xlsx` | One row per participant × window |
| `participant_summaries.xlsx` | One row per participant (means over good windows) |
| `rr_exports/<id>_segment_N.csv` | Raw / corrected RR + artifact flags per window |
| `config_used.json` | Exact config used (reproducibility) |
| `skipped_segments.csv` | Segment mode only: listed segments that could not be analysed, with the reason |

---

## Methods & references

- Task Force of ESC & NASPE. Heart rate variability: standards of measurement.
  *Circulation* 1996;93:1043–1065.
- Tarvainen MP et al. An advanced detrending method with application to HRV analysis.
  *IEEE Trans Biomed Eng* 2002;49(2):172–175.
- Lipponen JA, Tarvainen MP. A robust algorithm for HRV time series artefact correction.
  *J Med Eng Technol* 2019;43(3):173–181.
- Kubios HRV Scientific Edition — preprocessing documentation, 2024.

---

## Project structure

```
nm_hrv/
├── nm_hrv/
│   ├── __init__.py        # Package metadata
│   ├── loaders.py         # EDF / WFDB 
│   ├── preprocessing.py   # Bandpass, notch, normalisation
│   ├── sqi.py             # Signal quality metrics & classification
│   ├── rr.py              # R-peak detection & artifact correction
│   ├── hrv.py             # Time / frequency / nonlinear HRV
│   ├── summaries.py       # Record & cohort aggregation
│   ├── pipeline.py        # Orchestration
│   └── utils.py           # Age helpers, smoothness-priors detrend
│
├── configs/
│   ├── edf_config.yaml
│   └── wfdb_config.yaml
│
├── run_pipeline.py
├── requirements.txt
├── setup.py
└── README.md
```

---

