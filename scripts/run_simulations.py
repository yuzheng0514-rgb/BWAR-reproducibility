#!/usr/bin/env python3
"""Run the fixed-reference geometry study reported as Simulation 1."""

from __future__ import annotations

import argparse
from pathlib import Path

from bwar.paper_jcgs import build_strong_synthetic_artifacts as simulation


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Bures, Log-Euclidean, and Cholesky chart-alignment "
            "study and the Bures finite-sample variations."
        )
    )
    parser.add_argument("--reps", type=int, default=80)
    parser.add_argument("--ar-model", choices=("diag", "full"), default="full")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=simulation.ROOT / "results" / "generated" / "s1_geometry",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=simulation.ROOT / "artifacts" / "generated",
    )
    args = parser.parse_args()

    simulation.OUT_DIR = args.output_root
    simulation.OVERLEAF = args.artifact_root
    simulation.TABLE_DIR = args.artifact_root / "tables"
    simulation.FIGURE_DIR = args.artifact_root / "figures"
    simulation.OUT_DIR.mkdir(parents=True, exist_ok=True)
    simulation.TABLE_DIR.mkdir(parents=True, exist_ok=True)
    simulation.FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    raw, summary = simulation.run(args.reps, ar_model=args.ar_model)
    raw.to_csv(
        simulation.OUT_DIR / "strong_synthetic_transport_raw.csv", index=False
    )
    summary.to_csv(
        simulation.OUT_DIR / "strong_synthetic_transport_summary.csv", index=False
    )
    simulation.write_main_table(
        summary, simulation.TABLE_DIR / "synthetic_transport_main.tex"
    )
    simulation.write_variation_table(
        summary, simulation.TABLE_DIR / "synthetic_transport_variation.tex"
    )
    simulation.make_figure(
        raw,
        summary,
        simulation.FIGURE_DIR / "synthetic_transport_mechanism",
    )
    print(f"S1 simulation results: {args.output_root}")
    print(f"Generated tables and figures: {args.artifact_root}")


if __name__ == "__main__":
    main()
