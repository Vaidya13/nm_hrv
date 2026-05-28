"""
loaders.py
----------
ECG data loaders for EDF, WFDB, and compressed waveform table formats.

Provides
--------
  load_edf()         MNE-based EDF/BDF reader with auto ECG channel selection
  load_wfdb()        WFDB (PhysioNet) record reader
  load_waves()       Compressed int16 waveform table row (zlib + base64)
  load_ecg_record()  Unified dispatcher
  find_ecg_channel() Automatic best-channel selection from multi-channel EDF
  merge_metadata()   Join loader output with external metadata CSV
"""

import zlib
import base64

import mne
import wfdb
import numpy as np
import pandas as pd

from pathlib import Path

import neurokit2 as nk

from .sqi import compute_sqi


# ===========================================================================
# AUTO ECG CHANNEL SELECTION
# ===========================================================================

# Channels to unconditionally skip
_EXCLUDE_KEYWORDS = [
    "acc", "accelerometer", "gyro", "ppg", "resp",
    "eda", "emg", "eeg", "spo2", "temp", "gsr",
    "activity", "steps", "pressure",
]

# Preferred ECG channel name fragments (ordered by priority)
_CANDIDATE_KEYWORDS = [
    "ecg:gel", "ecg:dry", "ecg", "ekg",
    "lead", "mlii", "avf", "avl", "avr",
    "ii", "i", "v1", "v5", "chest",
]


def find_ecg_channel(raw):
    """
    Select the best ECG channel from a multi-channel MNE Raw object.

    Strategy
    --------
    1. Exclude obvious non-ECG channel types (accelerometer, EEG, etc.)
    2. Keep channels whose names contain ECG-like keywords.
    3. Score each candidate on:
         - QRS band power ratio  (5 weight)
         - Signal variance
         - HR plausibility bonus (10 weight)
    4. Return the highest-scoring channel.

    Raises
    ------
    ValueError
        If no ECG-like channel is found, or none is usable.
    """
    candidate_channels = []

    for ch in raw.ch_names:
        ch_lower = ch.lower()

        if any(k in ch_lower for k in _EXCLUDE_KEYWORDS):
            continue

        if any(k in ch_lower for k in _CANDIDATE_KEYWORDS):
            candidate_channels.append(ch)

    if not candidate_channels:
        raise ValueError(
            "No ECG-like channel found in file. "
            f"Available channels: {raw.ch_names}"
        )

    best_score   = -np.inf
    best_channel = None

    for ch in candidate_channels:
        try:
            sig = raw.get_data(picks=[ch])[0]
            fs  = float(raw.info["sfreq"])

            # Amplitude scale guard (convert V → mV if needed)
            if np.nanstd(sig) < 0.01:
                sig = sig * 1000.0

            variance  = np.nanvar(sig)
            sqi       = compute_sqi(sig, fs)
            qrs_score = sqi.get("qrs_power_ratio", 0.0) or 0.0

            # Quick HR plausibility from detected peaks
            cleaned   = nk.ecg_clean(sig, sampling_rate=fs)
            _, info   = nk.ecg_peaks(
                cleaned,
                sampling_rate=fs,
                method="pantompkins1985",
            )
            rpeaks      = info["ECG_R_Peaks"]
            duration    = len(sig) / fs
            hr          = (len(rpeaks) / duration) * 60.0 if duration > 0 else 0.0
            hr_score    = 1.0 if 35 <= hr <= 180 else 0.0

            score = qrs_score * 5 + variance + hr_score * 10

            if score > best_score:
                best_score   = score
                best_channel = ch

        except Exception:
            continue

    if best_channel is None:
        raise ValueError(
            "No usable ECG channel found after scoring. "
            "Check signal quality or channel names."
        )

    print(f"[loaders] Selected ECG channel: {best_channel} (score={best_score:.3f})")
    return best_channel


# ===========================================================================
# METADATA MERGER
# ===========================================================================

def merge_metadata(row_dict, df_metadata):
    """
    Merge a loader output dict with a participant metadata DataFrame.

    The metadata CSV must have a 'participant_id' column.
    Metadata fields are added to row_dict; existing keys are NOT overwritten.

    Parameters
    ----------
    row_dict : dict
    df_metadata : pd.DataFrame or None

    Returns
    -------
    pd.Series
    """
    if df_metadata is None:
        return pd.Series(row_dict)

    participant_id = str(row_dict["participant_id"])
    match = df_metadata[
        df_metadata["participant_id"].astype(str) == participant_id
    ]

    if len(match) > 0:
        for col, val in match.iloc[0].items():
            if col not in row_dict:          # never overwrite loader fields
                row_dict[col] = val

    return pd.Series(row_dict)


# ===========================================================================
# EDF LOADER
# ===========================================================================

def load_edf(edf_file, df_metadata=None):
    """
    Load an EDF/BDF file and return a participant record Series.

    Parameters
    ----------
    edf_file : str or Path
        Path to .edf file.
    df_metadata : pd.DataFrame, optional
        Metadata table with 'participant_id' column.

    Returns
    -------
    pd.Series
        Fields: participant_id, frequency, ecg_channel, ecg, source_format,
                + any metadata columns.
    """
    raw = mne.io.read_raw_edf(
        str(edf_file),
        preload=True,
        verbose=False,
    )

    fs          = float(raw.info["sfreq"])
    ecg_channel = find_ecg_channel(raw)
    ecg         = raw.get_data(picks=[ecg_channel])[0]

    # Unit guard: convert V → mV if std < 0.01 (typical raw EDF in volts)
    if np.nanstd(ecg) < 0.01:
        ecg = ecg * 1000.0

    participant_id = Path(edf_file).stem

    row_dict = {
        "participant_id": participant_id,
        "frequency":      fs,
        "ecg_channel":    ecg_channel,
        "ecg":            ecg,
        "source_format":  "edf",
    }

    return merge_metadata(row_dict, df_metadata)


# ===========================================================================
# WFDB LOADER  (PhysioNet / MIT-BIH etc.)
# ===========================================================================

def load_wfdb(record_path, df_metadata=None):
    """
    Load a WFDB record (PhysioNet format).

    Reads the first signal channel; override via config if needed.

    Parameters
    ----------
    record_path : str
        Path prefix (without extension) to .hea / .dat files.
    df_metadata : pd.DataFrame, optional

    Returns
    -------
    pd.Series
    """
    record = wfdb.rdrecord(str(record_path))
    fs     = float(record.fs)

    # Use first channel; could be extended to channel selection
    ecg = record.p_signal[:, 0].astype(np.float64)

    participant_id = Path(record_path).stem

    row_dict = {
        "participant_id": participant_id,
        "frequency":      fs,
        "ecg":            ecg,
        "source_format":  "wfdb",
    }

    return merge_metadata(row_dict, df_metadata)


# ===========================================================================
# WAVEFORM TABLE LOADER  (compressed int16 rows)
# ===========================================================================

def decode_ecg_waveform(encoded, gain):
    """
    Decode a zlib-compressed, base64-encoded int16 ECG waveform.

    Parameters
    ----------
    encoded : str or bytes
        Base64-encoded, zlib-compressed int16 ECG bytes.
    gain : float
        ADC gain to convert raw int16 → physical units.

    Returns
    -------
    np.ndarray (float32)
    """
    if isinstance(encoded, str):
        encoded = encoded.encode("utf-8")

    compressed  = base64.b64decode(encoded)
    decompressed = zlib.decompress(compressed)
    ecg         = np.frombuffer(decompressed, dtype=np.int16).astype(np.float32)
    ecg         = ecg * float(gain)

    return ecg


def load_waves(row):
    """
    Load one row from a waveform table (DataFrame row with 'waveform' column).

    Expected columns: participant_id, frequency, gain, waveform,
                      + any additional metadata columns.

    Returns
    -------
    pd.Series
    """
    ecg = decode_ecg_waveform(row["waveform"], row["gain"])

    row_dict = {
        "participant_id": str(row["participant_id"]),
        "frequency":      float(row["frequency"]),
        "ecg":            ecg,
        "source_format":  "waves",
    }

    # Carry over all other columns except the raw waveform blob
    for col in row.index:
        if col not in ("waveform", "gain") and col not in row_dict:
            row_dict[col] = row[col]

    return pd.Series(row_dict)


# ===========================================================================
# UNIFIED DISPATCHER
# ===========================================================================

def load_ecg_record(file, data_format, df_metadata=None):
    """
    Load an ECG record from any supported format.

    Parameters
    ----------
    file : str or Path
        File path (for edf/wfdb) or WFDB record prefix.
    data_format : str
        One of: "edf", "wfdb".
    df_metadata : pd.DataFrame, optional

    Returns
    -------
    pd.Series
        Standardised participant record.

    Raises
    ------
    ValueError
        For unsupported data formats.
    """
    if data_format == "edf":
        return load_edf(file, df_metadata)

    elif data_format == "wfdb":
        return load_wfdb(file, df_metadata)

    else:
        raise ValueError(
            f"Unsupported data format: '{data_format}'. "
            "Supported: 'edf', 'wfdb'."
        )