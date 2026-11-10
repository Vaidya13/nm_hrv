"""
segments.py
-----------
Analyse only chosen parts of a recording (e.g. resting state inside a longer
recording that also contains a task).

The collaborator supplies a CSV (config key ``segments_csv``) with one row per
segment to analyse:

    participant_id,start_sec,stop_sec,label
    P001,120,480,rest
    P001,1500,1700,rest_2
    P002,00:03:00,00:08:30,rest

  participant_id   must match the recording file name without extension
  start_sec        segment start, measured from the START OF THE RECORDING.
                   Seconds (e.g. 120 or 120.5) or clock-style text
                   MM:SS / HH:MM:SS(.fff)
  stop_sec         segment end, same format
  any other column (e.g. label, condition) is copied into every output row

Accepted header aliases: start / start_time, stop / end / end_sec / stop_time.

Window planning (``plan_windows``)
----------------------------------
Segment shorter than ``min_window_sec``      -> not analysed; listed in
                                                 skipped_segments.csv with the reason
segment_windowing = "whole"                   -> the segment is ONE window of
                                                 whatever length it has
segment_windowing = "tile"  (default)         -> consecutive windows of
                                                 ``window_sec``; a final partial
                                                 window is kept if it is at least
                                                 ``min_window_sec`` long. A segment
                                                 shorter than ``window_sec`` becomes
                                                 a single shorter window.
"""

import re

import numpy as np
import pandas as pd


_START_ALIASES = ("start_sec", "start", "start_time")
_STOP_ALIASES  = ("stop_sec", "stop", "end", "end_sec", "stop_time")
_CORE_COLUMNS  = {"participant_id", "start_sec", "stop_sec"}


# ===========================================================================
# PARSING
# ===========================================================================

def parse_time(value):
    """
    Convert a time value to seconds.

    Accepts numbers, numeric strings ("120", "120.5") and clock-style strings
    ("MM:SS", "HH:MM:SS", "HH:MM:SS.fff"). Raises ValueError otherwise.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        raise ValueError("empty time value")

    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)

    text = str(value).strip()
    if text == "":
        raise ValueError("empty time value")

    if re.fullmatch(r"[+-]?\d+(\.\d+)?", text):
        return float(text)

    parts = text.split(":")
    if len(parts) in (2, 3) and all(re.fullmatch(r"\d+(\.\d+)?", p) for p in parts):
        nums = [float(p) for p in parts]
        if len(nums) == 2:
            return nums[0] * 60.0 + nums[1]
        return nums[0] * 3600.0 + nums[1] * 60.0 + nums[2]

    raise ValueError(f"cannot read time value '{value}' "
                     "(use seconds, MM:SS or HH:MM:SS)")


def load_segments_csv(path):
    """
    Read and validate a segments CSV.

    Returns
    -------
    pd.DataFrame
        Columns: participant_id (str), start_sec (float), stop_sec (float),
        plus any extra columns. Rows are sorted by participant and start.

    Raises
    ------
    ValueError
        With a message naming the offending CSV rows (1-based, header = row 1).
    """
    df = pd.read_csv(path, dtype={"participant_id": str})
    df.columns = [str(c).strip() for c in df.columns]

    if "participant_id" not in df.columns:
        raise ValueError("segments CSV needs a 'participant_id' column")

    def _find(aliases):
        for a in aliases:
            if a in df.columns:
                return a
        return None

    c_start, c_stop = _find(_START_ALIASES), _find(_STOP_ALIASES)
    if c_start is None or c_stop is None:
        raise ValueError(
            "segments CSV needs a start column "
            f"({' / '.join(_START_ALIASES)}) and a stop column "
            f"({' / '.join(_STOP_ALIASES)})"
        )

    df["participant_id"] = df["participant_id"].astype(str).str.strip()

    problems = []
    starts, stops = [], []
    for i, (a, b) in enumerate(zip(df[c_start], df[c_stop])):
        try:
            s, e = parse_time(a), parse_time(b)
        except ValueError as exc:
            problems.append(f"row {i + 2}: {exc}")
            starts.append(np.nan); stops.append(np.nan)
            continue
        if e <= s:
            problems.append(f"row {i + 2}: stop ({e:g}) is not after start ({s:g})")
        if s < 0:
            problems.append(f"row {i + 2}: negative start ({s:g})")
        starts.append(s); stops.append(e)

    if problems:
        raise ValueError("problems in segments CSV:\n  " + "\n  ".join(problems[:20]))

    df = df.drop(columns=[c for c in (c_start, c_stop) if c not in _CORE_COLUMNS])
    df["start_sec"] = starts
    df["stop_sec"]  = stops

    return df.sort_values(["participant_id", "start_sec"]).reset_index(drop=True)


def find_overlaps(df_seg):
    """Return a list of (participant_id, row_a, row_b) for overlapping segments."""
    out = []
    for pid, g in df_seg.groupby("participant_id"):
        g = g.sort_values("start_sec")
        prev_stop, prev_idx = None, None
        for idx, r in g.iterrows():
            if prev_stop is not None and r["start_sec"] < prev_stop:
                out.append((pid, prev_idx + 2, idx + 2))
            if prev_stop is None or r["stop_sec"] > prev_stop:
                prev_stop, prev_idx = r["stop_sec"], idx
    return out


# ===========================================================================
# WINDOW PLANNING
# ===========================================================================

def plan_windows(segs, fs, n_samples, config):
    """
    Turn the segment rows of ONE participant into analysis windows.

    Parameters
    ----------
    segs : pd.DataFrame
        Rows of the segments table for this participant.
    fs : float
        Sampling rate of the recording.
    n_samples : int
        Length of the recording in samples.
    config : dict
        Uses window_sec, min_window_sec (default 60), segment_windowing
        ("tile" default, or "whole").

    Returns
    -------
    windows : list of dict
        start, stop (sample indices), meta (dict of scalar output columns)
    skipped : list of dict
        meta (dict) and failure_reason, for segments that could not be used
    """
    window_sec = float(config["window_sec"])
    min_sec    = float(config.get("min_window_sec", 60))
    mode       = config.get("segment_windowing", "tile")
    rec_dur    = n_samples / fs

    if mode not in ("tile", "whole"):
        raise ValueError("segment_windowing must be 'tile' or 'whole'")

    extras  = [c for c in segs.columns if c not in _CORE_COLUMNS]
    windows, skipped = [], []

    for seg_id, (_, s) in enumerate(segs.iterrows()):
        t0, t1 = float(s["start_sec"]), float(s["stop_sec"])

        meta = {
            "segment_id":        seg_id,
            "segment_start_sec": t0,
            "segment_stop_sec":  t1,
        }
        for c in extras:
            val = s[c]
            if isinstance(val, (str, int, float, bool, np.integer, np.floating)):
                meta[c] = val

        # clip to the recording
        c0, c1 = max(0.0, t0), min(rec_dur, t1)
        dur    = c1 - c0
        meta["segment_analysed_sec"] = max(0.0, dur)

        if c1 <= c0:
            skipped.append({"meta": meta, "failure_reason": "segment_outside_recording"})
            continue
        if dur < min_sec:
            reason = ("segment_too_short" if t1 <= rec_dur and t0 >= 0
                      else "segment_too_short_after_clipping")
            skipped.append({"meta": meta, "failure_reason": reason})
            continue

        if mode == "whole":
            spans = [(c0, dur)]
        else:
            n_full = int(dur // window_sec)
            if n_full == 0:
                spans = [(c0, dur)]
            else:
                spans = [(c0 + k * window_sec, window_sec) for k in range(n_full)]
                rem = dur - n_full * window_sec
                if rem >= min_sec:
                    spans.append((c0 + n_full * window_sec, rem))

        for k, (a, length) in enumerate(spans):
            w = dict(meta)
            w["window_in_segment"] = k
            windows.append({
                "start": int(round(a * fs)),
                "stop":  min(n_samples, int(round((a + length) * fs))),
                "meta":  w,
            })

    return windows, skipped
