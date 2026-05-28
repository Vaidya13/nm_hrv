"""
Kubios-benchmarked HRV harmonization framework

Supports:
  - Age-adaptive artifact correction (Lipponen & Tarvainen 2019)
  - Cubic spline interpolation + 4 Hz resampling
  - Smoothness-priors detrending (Tarvainen et al. 2002)
  - Welch PSD (Task Force / Kubios standard)
  - Hierarchical SQI scoring
  - HRV confidence scoring
  - Cohort-level summaries
"""

__version__ = "1.0"
__author__  = "Nilakshi Vaidya"