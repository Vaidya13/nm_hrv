# HRV-harmonization-pipeline
Kubios-benchmarked HRV harmonization pipeline

A Python package for large-scale ECG heart rate variability (HRV)
analysis across diverse cohorts (adults, adolescents, children).

It implements the Task Force (1996) / Kubios standard frequency-domain pipeline:

1. ECG preprocessing (bandpass + notch + robust normalisation)
2. R-peak detection (Pan-Tompkins via neurokit2)
3. Artifact classification with adaptive Lipponen & Tarvainen (2019) thresholds
4. Cubic spline interpolation of artifact beats
5. 4 Hz resampling onto a regular time grid
6. Smoothness-priors detrending (Tarvainen et al. 2002)
7. Welch PSD — 8 windows, 50 % overlap, Hamming taper
8. LF / HF / VLF power extraction (Task Force bands)
9. Nonlinear HRV (SD1, SD2, SD1/SD2)
10. Hierarchical SQI classification + HRV confidence score
11. Cohort-level summaries with usable-fraction QC


## Project structure

```
nm_hrv/
├── nm_hrv/
│   ├── __init__.py        # Package metadata
│   ├── loaders.py         # EDF / WFDB / waveform loaders
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


# Installation

pip install -r requirements.txt

---

# Example Usage

```bash
# EDF cohort
python run_pipeline.py --config configs/edf_config.yaml

# PhysioNet / WFDB cohort
python run_pipeline.py --config configs/wfdb_config.yaml
```

Edit the relevant config YAML to point at your data folder and metadata CSV before running.

---

# Outputs

- all_hrv_results.xlsx
- participant_summaries.xlsx
- rr_exports/
- config_used.json








