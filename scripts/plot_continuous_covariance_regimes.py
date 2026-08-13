#!/usr/bin/env python3
"""Plot the two continuous covariance regimes as absolute test loss."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


METHOD_STYLES = (
    ("persistence", "Persistence", "#6E6E6E", ":", "x"),
    ("euclidean", "Euclidean AR", "#5B8DB8", "-", "o"),
    ("cholesky", "Cholesky AR", "#C28455", "-", "s"),
    ("log_euclidean", "log-Euclidean AR", "#8066A6", "-", "^"),
    ("fixed", "Fixed BWAR", "#2F6DB2", "--", "D"),
    ("local", "Local BWAR", "#C94B45", "-", "P"),
)


def _panel(axis: plt.Axes, frame: pd.DataFrame, horizon: int) -> list[plt.Line2D]:
    handles = []
    for method, label, color, linestyle, marker in METHOD_STYLES:
        cell = frame.loc[
            frame.horizon.eq(horizon) & frame.method.eq(method)
        ].sort_values("movement_control")
        emphasized = method in {"fixed", "local"}
        if emphasized:
            axis.fill_between(
                cell.movement_control.to_numpy(float),
                np.maximum(cell.test_w2_ci_low.to_numpy(float), 1e-12),
                np.maximum(cell.test_w2_ci_high.to_numpy(float), 1e-12),
                color=color,
                alpha=0.10,
                linewidth=0,
            )
        handles.append(
            axis.plot(
                cell.movement_control,
                cell.test_w2_mean,
                color=color,
                linestyle=linestyle,
                marker=marker,
                markersize=4.2 if emphasized else 3.7,
                markerfacecolor="white",
                markeredgewidth=0.85,
                linewidth=2.0 if emphasized else 0.9,
                alpha=1.0 if emphasized else 0.64,
                label=label,
                zorder=3,
            )[0]
        )
    axis.set_yscale("log")
    axis.grid(axis="y", which="both", color="#E6E6E6", linewidth=0.5)
    return handles


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--output-stem", type=Path, required=True)
    args = parser.parse_args()

    summary = pd.read_csv(args.result_dir / "summary.csv")
    horizons = (1, 3, 6)
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.0,
            "axes.labelsize": 7.3,
            "axes.titlesize": 8.0,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    figure, axes = plt.subplots(2, 3, figsize=(7.25, 4.75), facecolor="white")
    handles = []
    rows = (
        (
            "persistent",
            "(a) Smooth center displacement",
            r"Center-path displacement, $\Delta$",
        ),
        (
            "mean_reverting",
            "(b) Fixed-center multiscale cycle",
            "Cycle amplitude multiplier",
        ),
    )
    for row, (regime, row_title, x_label) in enumerate(rows):
        row_frame = summary.loc[summary.regime.eq(regime)]
        ticks = sorted(row_frame.movement_control.unique())
        for column, horizon in enumerate(horizons):
            panel_handles = _panel(axes[row, column], row_frame, horizon)
            if row == 0 and column == 0:
                handles = panel_handles
            axes[row, column].set_title(rf"$h={horizon}$", pad=4)
            axes[row, column].set_xticks(ticks)
            axes[row, column].set_xlabel(x_label)
            if column == 0:
                axes[row, column].set_ylabel(
                    r"Mean Gaussian $W_2^2$ test loss (log scale)"
                )
    figure.legend(
        handles=handles,
        labels=[handle.get_label() for handle in handles],
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.012),
        fontsize=6.7,
        handlelength=2.8,
        columnspacing=1.35,
        frameon=False,
    )
    figure.subplots_adjust(
        left=0.105,
        right=0.99,
        bottom=0.205,
        top=0.87,
        hspace=0.72,
        wspace=0.22,
    )
    for row, (_, row_title, _) in enumerate(rows):
        row_box = axes[row, 0].get_position()
        figure.text(
            axes[row, 0].get_position().x0,
            0.965 if row == 0 else row_box.y1 + 0.045,
            row_title,
            fontsize=7.4,
            fontweight="bold",
            ha="left",
            va="center",
        )

    args.output_stem.parent.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for extension, kwargs in (
        ("pdf", {}),
        ("svg", {}),
        ("png", {"dpi": 300}),
        (
            "tiff",
            {
                "dpi": 600,
                "transparent": False,
                "pil_kwargs": {"compression": "tiff_lzw"},
            },
        ),
    ):
        output = args.output_stem.with_suffix(f".{extension}")
        figure.savefig(output, facecolor="white", **kwargs)
        if extension == "tiff":
            with Image.open(output) as raster:
                rgb = raster.convert("RGB")
                rgb.save(
                    output,
                    format="TIFF",
                    compression="tiff_lzw",
                    dpi=(600, 600),
                )
        outputs[extension] = str(output.resolve())
    plt.close(figure)
    manifest = {
        "source_summary": str((args.result_dir / "summary.csv").resolve()),
        "result_dir": str(args.result_dir.resolve()),
        "layout": "2x3 absolute-loss line figure",
        "uncertainty": "95% normal Monte Carlo intervals across paired replications",
        "transformations": [
            "replication means grouped by regime, control, horizon, and method",
            "logarithmic y-axis without smoothing or filtering",
        ],
        "outputs": outputs,
    }
    args.output_stem.with_suffix(".json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
