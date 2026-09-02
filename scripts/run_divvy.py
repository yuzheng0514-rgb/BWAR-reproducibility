#!/usr/bin/env python3
"""Run the complete rolling-origin Divvy analysis."""

from __future__ import annotations

import argparse
from pathlib import Path

from bwar.paper_jcgs import run_divvy_analysis as divvy


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download/prepare the Divvy data and reproduce the rolling-origin "
            "analysis, local-reference tuning, and paired inference."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=divvy.ROOT / "results" / "generated" / "divvy",
    )
    parser.add_argument(
        "--processed-data-dir",
        type=Path,
        help="Use the submitted Divvy processed-data directory instead of raw archives.",
    )
    args = parser.parse_args()
    divvy.run(args.output_dir, processed_data_dir=args.processed_data_dir)


if __name__ == "__main__":
    main()
