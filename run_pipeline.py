"""
run_pipeline.py
---------------
Command-line entry point for the nm_hrv pipeline.

Usage
-----
    python run_pipeline.py --config configs/edf_config.yaml
    python run_pipeline.py --config configs/wfdb_config.yaml

Or via the installed console script:
    nm_hrv --config configs/edf_config.yaml
"""

import argparse
import sys
import yaml

from nm_hrv.pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="nm_hrv – HRV preprocessing pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py --config configs/edf_config.yaml
  python run_pipeline.py --config configs/wfdb_config.yaml
        """,
    )
    parser.add_argument(
        "--config",
        required=True,
        metavar="PATH",
        help="Path to YAML configuration file",
    )

    args = parser.parse_args()

    try:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"[error] Config file not found: {args.config}", file=sys.stderr)
        sys.exit(1)
    except yaml.YAMLError as exc:
        print(f"[error] Invalid YAML in {args.config}: {exc}", file=sys.stderr)
        sys.exit(1)

    run_pipeline(config)


if __name__ == "__main__":
    main()