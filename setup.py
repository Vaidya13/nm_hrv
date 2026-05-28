from setuptools import setup, find_packages

setup(
    name="nm_hrv",
    version="1.0",
    author="Nilakshi Vaidya",
    description=(
        "Kubios-benchmarked HRV harmonization pipeline "
        "for EDF, WFDB, and waveform table inputs. "
        "Supports age-adaptive artifact correction, 4 Hz cubic spline resampling, "
        "smoothness-priors detrending, and Welch PSD."
    ),
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "matplotlib",
        "neurokit2",
        "mne",
        "wfdb",
        "pyyaml",
        "openpyxl",
    ],
    entry_points={
        "console_scripts": [
            "nm_hrv=run_pipeline:main",
        ],
    },
)