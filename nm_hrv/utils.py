"""
utils.py
--------
Age-adaptive helpers shared across modules.
"""

import numpy as np
import scipy.sparse
import scipy.sparse.linalg


# ---------------------------------------------------------------------------
# Age grouping
# ---------------------------------------------------------------------------

def get_age_group(age):
    """Return a human-readable age group label."""
    if age is None or (isinstance(age, float) and np.isnan(age)):
        return "unknown"
    if age < 13:
        return "child"
    elif age < 18:
        return "adolescent"
    else:
        return "adult"


# ---------------------------------------------------------------------------
# Age-adaptive RR bounds (physiological plausibility)
# ---------------------------------------------------------------------------

def get_age_adaptive_rr_bounds(age):
    """
    Return (rr_min_ms, rr_max_ms) physiological hard limits.

    References
    ----------
    Task Force 1996; Massin et al. 2000 (paediatric norms)
    """
    if age is None or (isinstance(age, float) and np.isnan(age)):
        return (300, 2000)

    if age < 13:          # child
        return (250, 1500)
    elif age < 18:        # adolescent
        return (300, 1800)
    else:                 # adult
        return (300, 2000)


# ---------------------------------------------------------------------------
# Age-adaptive HR plausibility bounds
# ---------------------------------------------------------------------------

def get_age_adaptive_hr_bounds(age):
    """Return (hr_min_bpm, hr_max_bpm) plausibility bounds."""
    if age is None or (isinstance(age, float) and np.isnan(age)):
        return (30, 220)

    if age < 13:
        return (50, 220)
    elif age < 18:
        return (40, 210)
    else:
        return (30, 200)


# ---------------------------------------------------------------------------
# Smoothness-priors detrending  (Tarvainen et al. 2002, IEEE TBME)
# ---------------------------------------------------------------------------

def smoothness_priors_detrend(rr, lambda_val=300):
    """
    Remove slow non-stationary trends using the smoothness-priors method.

    This is the default detrending used in Kubios HRV. It acts as a
    time-varying FIR high-pass filter without requiring regular sampling.

    Implementation uses a **sparse** second-order difference matrix so the
    solver scales to overnight PSG recordings (N > 100 000 samples) without
    memory or speed issues.  A dense N×N approach would require ~1 GB RAM
    for a 1-hour 4 Hz signal; the sparse approach uses < 1 MB.

    Parameters
    ----------
    rr : array_like
        RR interval series (ms).  Works on both beat-indexed and regularly
        resampled series.
    lambda_val : float
        Regularisation parameter.
        Kubios default = 300 for 5-min short-term HRV.
        Use 500–1000 for longer recordings or children (higher HRV).
        Larger → more aggressive low-frequency removal.

    Returns
    -------
    rr_detrended : np.ndarray
        Trend-free RR series, same length as input.

    References
    ----------
    Tarvainen MP et al. IEEE Trans Biomed Eng. 2002;49(2):172-175.
    """
    rr = np.asarray(rr, dtype=float)
    N  = len(rr)

    if N < 4:
        return rr - np.mean(rr)

    # ---- Sparse second-order difference matrix D2 (shape: N-2 × N) ---------
    # D2 = diff(I, n=2) built directly as a sparse diagonal matrix.
    # Rows: [1, -2, 1] stencil at each interior position.
    ones  = np.ones(N - 2)
    diags = [ones, -2 * ones, ones]
    D2    = scipy.sparse.diags(diags, offsets=[0, 1, 2], shape=(N - 2, N),
                               format="csc")

    # ---- Regularised system: (I + λ² D2ᵀ D2) trend = rr ------------------
    I_sp       = scipy.sparse.eye(N, format="csc")
    reg_matrix = I_sp + (lambda_val ** 2) * (D2.T @ D2)

    try:
        trend = scipy.sparse.linalg.spsolve(reg_matrix, rr)
    except Exception:
        # Fallback: simple linear detrend (never happens with well-formed input)
        from scipy.signal import detrend as scipy_detrend
        return scipy_detrend(rr, type="linear")

    return rr - trend